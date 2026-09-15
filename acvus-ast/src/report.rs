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

pub struct Report<'a> {
    pub severity: Severity,
    pub message: String,
    pub path: &'a str,
    pub source: &'a str,
    pub span: Option<Span>,
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
        let text = index.line_text(self.source, start.line);
        let width = start.line.to_string().len();
        let underline = span
            .end
            .saturating_sub(span.start)
            .max(1)
            .min(text.len() + 1 - start.col.min(text.len() + 1));
        writeln!(f, "{:width$} |", "", width = width)?;
        writeln!(f, "{} | {}", start.line, text)?;
        writeln!(
            f,
            "{:width$} | {}{}",
            "",
            " ".repeat(start.col - 1),
            "^".repeat(underline.max(1)),
            width = width
        )
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
        };
        assert_eq!(
            report.to_string(),
            "error: no such thing\n  --> a.acvus:2:9\n  |\n2 | let n = len(d);\n  |         ^^^^^^\n"
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
