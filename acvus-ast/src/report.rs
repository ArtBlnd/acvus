//! A diagnostic rendered at a span of a source (RFC-0031): the message,
//! the file, the line and column, and the line itself with the span
//! marked.

use std::fmt;

use crate::span::Span;

/// Byte offsets of the start of every line of a source.
pub struct LineIndex {
    line_starts: Vec<usize>,
}

/// A 1-based line and column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LineCol {
    pub line: usize,
    pub col: usize,
}

impl LineIndex {
    pub fn new(source: &str) -> Self {
        let line_starts = std::iter::once(0)
            .chain(source.match_indices('\n').map(|(i, _)| i + 1))
            .collect();
        Self { line_starts }
    }

    pub fn line_col(&self, offset: usize) -> LineCol {
        let line = self.line_starts.partition_point(|&start| start <= offset);
        let line_start = self.line_starts[line - 1];
        LineCol {
            line,
            col: offset - line_start + 1,
        }
    }

    fn line_text<'s>(&self, source: &'s str, line: usize) -> &'s str {
        let start = self.line_starts[line - 1];
        let end = self
            .line_starts
            .get(line)
            .map_or(source.len(), |next| next - 1);
        &source[start..end.max(start)]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Severity::Error => write!(f, "error"),
            Severity::Warning => write!(f, "warning"),
        }
    }
}

/// A second place in the same diagnostic, with the words that say what
/// happened there. A label with no span is a note: it names no place, and
/// renders as `= help:` under the marked lines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Label {
    pub span: Option<Span>,
    pub text: String,
}

impl Label {
    pub fn at<S>(span: Span, text: S) -> Self
    where
        S: Into<String>,
    {
        Self {
            span: Some(span),
            text: text.into(),
        }
    }

    pub fn note<S>(text: S) -> Self
    where
        S: Into<String>,
    {
        Self {
            span: None,
            text: text.into(),
        }
    }
}

pub struct Report<'a> {
    pub severity: Severity,
    pub message: String,
    pub path: &'a str,
    pub source: &'a str,
    pub span: Option<Span>,
    pub labels: Vec<Label>,
}

/// One underlined span of the snippet: where it starts, how it is marked, and
/// the words that go after the marker.
struct Marked<'a> {
    span: Span,
    at: LineCol,
    head: char,
    text: &'a str,
}

impl Report<'_> {
    /// The spans this report marks, in source order, the primary first among
    /// spans that start together. The primary is underlined with `^` and a
    /// label with `-`; the primary carries the message where another span is
    /// marked too, so the reader can tell which line the message is about.
    fn marked(&self, primary: Span, index: &LineIndex) -> Vec<Marked<'_>> {
        let at = |span: Span| index.line_col(span.start.min(self.source.len()));
        let another_place = self.labels.iter().any(|l| l.span.is_some());
        let mut marked = vec![Marked {
            span: primary,
            at: at(primary),
            head: '^',
            text: match another_place {
                true => &self.message,
                false => "",
            },
        }];
        marked.extend(self.labels.iter().filter_map(|l| {
            l.span.map(|span| Marked {
                span,
                at: at(span),
                head: '-',
                text: &l.text,
            })
        }));
        marked.sort_by_key(|m| (m.span.start, m.head != '^'));
        marked
    }
}

fn write_marker(
    f: &mut fmt::Formatter<'_>,
    width: usize,
    line_text: &str,
    m: &Marked<'_>,
) -> fmt::Result {
    let underline = m
        .span
        .end
        .saturating_sub(m.span.start)
        .max(1)
        .min(line_text.len() + 1 - m.at.col.min(line_text.len() + 1))
        .max(1);
    write!(
        f,
        "{:width$} | {}{}",
        "",
        " ".repeat(m.at.col - 1),
        m.head.to_string().repeat(underline),
        width = width
    )?;
    match m.text.is_empty() {
        true => writeln!(f),
        false => writeln!(f, " {}", m.text),
    }
}

impl fmt::Display for Report<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}: {}", self.severity, self.message)?;
        let Some(span) = self.span else {
            return writeln!(f, "  --> {}", self.path);
        };
        let index = LineIndex::new(self.source);
        let start = index.line_col(span.start.min(self.source.len()));
        writeln!(f, "  --> {}:{}:{}", self.path, start.line, start.col)?;

        let marked = self.marked(span, &index);
        let width = marked
            .iter()
            .map(|m| m.at.line.to_string().len())
            .fold(start.line.to_string().len(), usize::max);

        writeln!(f, "{:width$} |", "", width = width)?;
        let mut shown: Option<usize> = None;
        for m in &marked {
            let line_text = index.line_text(self.source, m.at.line);
            match shown {
                Some(previous) if previous == m.at.line => {}
                Some(previous) => {
                    if m.at.line > previous + 1 {
                        writeln!(f, "...")?;
                    }
                    writeln!(f, "{:>width$} | {}", m.at.line, line_text, width = width)?;
                }
                None => writeln!(f, "{:>width$} | {}", m.at.line, line_text, width = width)?,
            }
            shown = Some(m.at.line);
            write_marker(f, width, line_text, m)?;
        }

        for note in self.labels.iter().filter(|l| l.span.is_none()) {
            writeln!(f, "{:width$} = help: {}", "", note.text, width = width)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_span_is_reported_at_its_line_and_column_with_the_line_shown() {
        let source = "let x = 1;\nlet n = len(d);\n";
        let report = Report {
            severity: Severity::Error,
            message: "no such thing".to_string(),
            path: "a.acvus",
            source,
            span: Some(Span::new(19, 25)),
            labels: Vec::new(),
        };
        assert_eq!(
            report.to_string(),
            "error: no such thing\n  --> a.acvus:2:9\n  |\n2 | let n = len(d);\n  |         ^^^^^^\n"
        );
    }

    #[test]
    fn a_label_on_an_earlier_line_is_marked_before_the_primary_span() {
        let source = "let a = [1, 2];\nlet b = a;\na\n";
        let report = Report {
            severity: Severity::Error,
            message: "`a` is used here after it was moved".to_string(),
            path: "mv.acvus",
            source,
            span: Some(Span::new(27, 28)),
            labels: vec![Label::at(Span::new(24, 25), "moved here")],
        };
        assert_eq!(
            report.to_string(),
            [
                "error: `a` is used here after it was moved",
                "  --> mv.acvus:3:1",
                "  |",
                "2 | let b = a;",
                "  |         - moved here",
                "3 | a",
                "  | ^ `a` is used here after it was moved",
                "",
            ]
            .join("\n")
        );
    }

    #[test]
    fn lines_between_two_marked_lines_are_elided_and_the_gutter_takes_the_widest() {
        let source = "a\nlet b = x;\nc\nd\ne\nf\ng\nh\ni\nlet e = x;\n";
        let report = Report {
            severity: Severity::Error,
            message: "twice".to_string(),
            path: "e.acvus",
            source,
            span: Some(Span::new(35, 36)),
            labels: vec![Label::at(Span::new(10, 11), "first here")],
        };
        assert_eq!(
            report.to_string(),
            [
                "error: twice",
                "  --> e.acvus:10:9",
                "   |",
                " 2 | let b = x;",
                "   |         - first here",
                "...",
                "10 | let e = x;",
                "   |         ^ twice",
                "",
            ]
            .join("\n")
        );
    }

    #[test]
    fn a_label_on_the_primary_line_gets_its_own_marker_and_a_spanless_one_is_a_help() {
        let source = "let x = v[0];\n";
        let report = Report {
            severity: Severity::Error,
            message: "cannot move out of index of `[String; 2]`".to_string(),
            path: "i.acvus",
            source,
            span: Some(Span::new(8, 12)),
            labels: vec![
                Label::at(Span::new(8, 9), "this is the container"),
                Label::note("borrow with `&v[i]`"),
            ],
        };
        assert_eq!(
            report.to_string(),
            [
                "error: cannot move out of index of `[String; 2]`",
                "  --> i.acvus:1:9",
                "  |",
                "1 | let x = v[0];",
                "  |         ^^^^ cannot move out of index of `[String; 2]`",
                "  |         - this is the container",
                "  = help: borrow with `&v[i]`",
                "",
            ]
            .join("\n")
        );
    }

    #[test]
    fn an_offset_at_the_end_of_the_source_is_on_the_last_line() {
        let index = LineIndex::new("ab\ncd");
        assert_eq!(index.line_col(5), LineCol { line: 2, col: 3 });
        assert_eq!(index.line_col(0), LineCol { line: 1, col: 1 });
        assert_eq!(index.line_col(3), LineCol { line: 2, col: 1 });
    }
}
