//! Synthesis boundaries retain Unicode offsets into the cleaned source. Reader
//! timing is measured per block, rather than stretching the whole document's
//! words over the currently generated audio.
pub(crate) struct SpeechBlock<'a> {
    pub text: &'a str,
    pub start: usize,
    pub end: usize,
    pub paragraph: usize,
}

pub(crate) fn speech_blocks(text: &str, target: usize) -> Vec<SpeechBlock<'_>> {
    let offsets = text
        .char_indices()
        .map(|(byte, _)| byte)
        .chain(std::iter::once(text.len()))
        .collect::<Vec<_>>();
    let chars = text.chars().collect::<Vec<_>>();
    let mut start = 0;
    let mut blocks = Vec::new();
    let target = target.clamp(1, 2000);
    while start < chars.len() {
        while start < chars.len() && chars[start].is_whitespace() {
            start += 1;
        }
        if start == chars.len() {
            break;
        }
        let limit = (start + target).min(chars.len());
        // Prefer paragraph and sentence endings. Long unbroken scripts still
        // make bounded progress without corrupting multi-byte characters.
        let end = if limit == chars.len() {
            limit
        } else {
            (start + target / 3..limit)
                .rev()
                .find(|&index| matches!(chars[index], '.' | '!' | '?' | '。' | '！' | '？' | '\n'))
                .map(|index| index + 1)
                .or_else(|| {
                    (start + 1..limit)
                        .rev()
                        .find(|&index| chars[index].is_whitespace())
                })
                .unwrap_or(limit)
        };
        let mut trimmed = end;
        while trimmed > start && chars[trimmed - 1].is_whitespace() {
            trimmed -= 1;
        }
        if trimmed > start {
            blocks.push(SpeechBlock {
                paragraph: 0,
                text: &text[offsets[start]..offsets[trimmed]],
                start,
                end: trimmed,
            });
        }
        start = end;
    }
    blocks
}

pub(crate) fn paragraph_blocks(text: &str, target: usize) -> Vec<SpeechBlock<'_>> {
    let mut blocks = Vec::new();
    let mut offset = 0;
    for (paragraph, line) in text.split_inclusive('\n').enumerate() {
        for mut block in speech_blocks(line.trim_end_matches('\n'), target) {
            block.start += offset;
            block.end += offset;
            block.paragraph = paragraph;
            blocks.push(block);
        }
        offset += line.chars().count();
    }
    blocks
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn paragraph_blocks_keep_long_paragraphs_together_and_preserve_unicode_offsets() {
        let text = format!(
            "  {}\n\n日本語の文章。🙂\r\nLast paragraph.",
            "A clear sentence. ".repeat(30)
        );
        let chars = text.chars().collect::<Vec<_>>();
        let blocks = paragraph_blocks(&text, 160);
        assert!(blocks.iter().filter(|block| block.paragraph == 0).count() > 2);
        assert_eq!(
            blocks
                .iter()
                .map(|block| block.paragraph)
                .collect::<std::collections::BTreeSet<_>>(),
            [0, 2, 3].into_iter().collect()
        );
        for block in &blocks {
            assert_eq!(
                block.text,
                chars[block.start..block.end].iter().collect::<String>()
            );
            assert!(!block.text.contains('\n'));
            assert!(block.end - block.start <= 160);
        }
        assert_eq!(
            blocks
                .iter()
                .flat_map(|block| block.text.chars())
                .filter(|c| !c.is_whitespace())
                .collect::<String>(),
            text.chars()
                .filter(|c| !c.is_whitespace())
                .collect::<String>()
        );
    }
    #[test]
    fn boundaries_preserve_content_offsets_and_unicode_under_every_size() {
        for text in [
            "  First sentence.\n\nSecond sentence!  ",
            "🙂日本語の文章。字字字\nNew paragraph.",
            "abcdefghijklmnopqrstuvwxyz",
        ] {
            for target in [1, 2, 7, 160] {
                let blocks = speech_blocks(text, target);
                let chars = text.chars().collect::<Vec<_>>();
                let rebuilt = blocks.iter().map(|block| block.text).collect::<String>();
                assert_eq!(
                    rebuilt
                        .chars()
                        .filter(|c| !c.is_whitespace())
                        .collect::<String>(),
                    text.chars()
                        .filter(|c| !c.is_whitespace())
                        .collect::<String>()
                );
                for block in blocks {
                    assert_eq!(
                        block.text,
                        chars[block.start..block.end].iter().collect::<String>()
                    );
                    assert!(block.end - block.start <= target);
                }
            }
        }
    }
}
