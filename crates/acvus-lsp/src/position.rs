//! Byte offsets, which the workspace speaks, against the protocol's line
//! and character positions.

use std::fmt;

use acvus_ast::Span;
use lsp_types::{Position, PositionEncodingKind, Range};

/// The unit a `Position`'s character counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    Utf8,
    Utf16,
}

impl Encoding {
    pub fn negotiate(offered: Option<&[PositionEncodingKind]>) -> Self {
        match offered.is_some_and(|offered| offered.contains(&PositionEncodingKind::UTF8)) {
            true => Encoding::Utf8,
            false => Encoding::Utf16,
        }
    }

    pub fn kind(self) -> PositionEncodingKind {
        match self {
            Encoding::Utf8 => PositionEncodingKind::UTF8,
            Encoding::Utf16 => PositionEncodingKind::UTF16,
        }
    }

    fn width(self, c: char) -> usize {
        match self {
            Encoding::Utf8 => c.len_utf8(),
            Encoding::Utf16 => c.len_utf16(),
        }
    }
}

/// The lines of one text, split at `\n`, `\r\n` and `\r` as the protocol
/// splits them.
pub struct LineIndex<'t> {
    text: &'t str,
    encoding: Encoding,
    lines: Vec<Line>,
}

#[derive(Debug, Clone, Copy)]
struct Line {
    start: usize,
    end_before_terminator: usize,
}

impl<'t> LineIndex<'t> {
    pub fn new(text: &'t str, encoding: Encoding) -> Self {
        let bytes = text.as_bytes();
        let mut lines = Vec::new();
        let mut start = 0;
        let mut at = 0;
        while at < bytes.len() {
            let terminator = match bytes[at] {
                b'\n' => 1,
                b'\r' if bytes.get(at + 1) == Some(&b'\n') => 2,
                b'\r' => 1,
                _ => {
                    at += 1;
                    continue;
                }
            };
            lines.push(Line {
                start,
                end_before_terminator: at,
            });
            at += terminator;
            start = at;
        }
        lines.push(Line {
            start,
            end_before_terminator: text.len(),
        });
        LineIndex {
            text,
            encoding,
            lines,
        }
    }

    pub fn position(&self, offset: usize) -> Result<Position, Unplaceable> {
        let unplaceable = Unplaceable {
            offset,
            len: self.text.len(),
        };
        if !self.text.is_char_boundary(offset) {
            return Err(unplaceable);
        }
        let number = self.lines.partition_point(|line| line.start <= offset) - 1;
        let line = self.lines[number];
        let character: usize = self.text[line.start..offset.min(line.end_before_terminator)]
            .chars()
            .map(|c| self.encoding.width(c))
            .sum();
        Ok(Position {
            line: u32::try_from(number).map_err(|_| unplaceable)?,
            character: u32::try_from(character).map_err(|_| unplaceable)?,
        })
    }

    /// The protocol clamps a line past the last to the text's end and a
    /// character past its line's end to that end.
    pub fn offset(&self, position: Position) -> usize {
        let past_the_last_line = self.text.len();
        let Ok(number) = usize::try_from(position.line) else {
            return past_the_last_line;
        };
        let Some(line) = self.lines.get(number) else {
            return past_the_last_line;
        };
        let Ok(character) = usize::try_from(position.character) else {
            return line.end_before_terminator;
        };
        let mut counted = 0;
        for (at, c) in self.text[line.start..line.end_before_terminator].char_indices() {
            let next = counted + self.encoding.width(c);
            if next > character {
                return line.start + at;
            }
            counted = next;
        }
        line.end_before_terminator
    }

    pub fn range(&self, span: Span) -> Result<Range, Unplaceable> {
        Ok(Range {
            start: self.position(span.start)?,
            end: self.position(span.end)?,
        })
    }

    pub fn span(&self, range: Range) -> Span {
        Span::new(self.offset(range.start), self.offset(range.end))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Unplaceable {
    pub offset: usize,
    pub len: usize,
}

impl fmt::Display for Unplaceable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "byte {} is no char boundary of the {}-byte text read now",
            self.offset, self.len
        )
    }
}

impl std::error::Error for Unplaceable {}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(line: u32, character: u32) -> Position {
        Position { line, character }
    }

    #[test]
    fn every_char_boundary_round_trips_in_both_encodings() {
        let text = "a\u{d55c}\u{1f600}b\r\nc\rd\n\n";
        for encoding in [Encoding::Utf8, Encoding::Utf16] {
            let index = LineIndex::new(text, encoding);
            for (offset, _) in text.char_indices().chain([(text.len(), ' ')]) {
                if text[..offset].ends_with('\r') && text[offset..].starts_with('\n') {
                    continue;
                }
                let position = index
                    .position(offset)
                    .expect("a char boundary is placeable");
                assert_eq!(index.offset(position), offset, "{encoding:?} {offset}");
            }
        }
    }

    #[test]
    fn a_char_counts_its_code_units() {
        let text = "\u{d55c}\u{1f600}x";
        let x = text.find('x').expect("the text holds x");
        assert_eq!(
            LineIndex::new(text, Encoding::Utf8).position(x),
            Ok(at(0, 7))
        );
        assert_eq!(
            LineIndex::new(text, Encoding::Utf16).position(x),
            Ok(at(0, 3))
        );
    }

    #[test]
    fn a_position_between_the_halves_of_a_char_is_its_start() {
        let text = "a\u{1f600}b";
        assert_eq!(LineIndex::new(text, Encoding::Utf16).offset(at(0, 2)), 1);
        assert_eq!(LineIndex::new(text, Encoding::Utf8).offset(at(0, 3)), 1);
    }

    #[test]
    fn an_offset_off_the_text_s_char_boundaries_is_unplaceable() {
        let text = "a\u{1f600}b";
        let index = LineIndex::new(text, Encoding::Utf8);
        assert_eq!(index.position(2), Err(Unplaceable { offset: 2, len: 6 }));
        assert_eq!(index.position(7), Err(Unplaceable { offset: 7, len: 6 }));
        assert_eq!(index.position(6), Ok(at(0, 6)));
    }

    #[test]
    fn a_position_past_its_line_or_the_text_is_clamped() {
        let text = "ab\r\ncd";
        let index = LineIndex::new(text, Encoding::Utf16);
        assert_eq!(index.offset(at(0, 9)), 2);
        assert_eq!(index.offset(at(1, 9)), text.len());
        assert_eq!(index.offset(at(7, 0)), text.len());
        assert_eq!(index.position(3), Ok(at(0, 2)));
    }

    #[test]
    fn an_offered_utf8_is_chosen() {
        let both = [PositionEncodingKind::UTF16, PositionEncodingKind::UTF8];
        assert_eq!(Encoding::negotiate(Some(&both)), Encoding::Utf8);
        assert_eq!(
            Encoding::negotiate(Some(&[PositionEncodingKind::UTF16])),
            Encoding::Utf16
        );
        assert_eq!(Encoding::negotiate(None), Encoding::Utf16);
    }
}
