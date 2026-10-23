//! Synthesis boundaries retain Unicode offsets into the cleaned source. Reader
//! timing is measured per block, rather than stretching the whole document's
//! words over the currently generated audio.
pub(crate) struct SpeechBlock<'a> {
    pub text: &'a str,
    pub start: usize,
    pub end: usize,
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
                text: &text[offsets[start]..offsets[trimmed]],
                start,
                end: trimmed,
            });
        }
        start = end;
    }
    blocks
}

#[cfg(test)]
mod tests {
    use super::*;
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
