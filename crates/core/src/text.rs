use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;
use unicode_normalization::UnicodeNormalization;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct TextCleaningOptions {
    pub enabled: bool,
    pub strip_markdown: bool,
    pub strip_html: bool,
    pub strip_code_blocks: bool,
    pub strip_special_characters: bool,
    pub normalize_whitespace: bool,
}

impl Default for TextCleaningOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            strip_markdown: true,
            strip_html: true,
            strip_code_blocks: true,
            strip_special_characters: true,
            normalize_whitespace: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CleanedText {
    pub text: String,
    pub source_format: &'static str,
    pub removed_code_blocks: usize,
    pub normalized_whitespace: bool,
}

static HTML_MARKER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)<(?:html|body|p|div|article|section|h[1-6]|ul|ol|li|br|table)\b").unwrap()
});
static HTML_HIDDEN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?is)<!--.*?-->|<(?:script|style|head|noscript|template|svg|canvas|iframe|object)\b[^>]*>.*?</(?:script|style|head|noscript|template|svg|canvas|iframe|object)\s*>",
    )
    .unwrap()
});
static HTML_BLOCKS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)</?(?:p|div|main|section|article|header|footer|nav|aside|h[1-6]|blockquote|pre|address|figure|figcaption|details|summary|fieldset|legend|dl|dt|dd|ul|ol|table|caption|form|li|tr|br|hr)\b[^>]*>").unwrap()
});
static HTML_CELLS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)</?(?:td|th)\b[^>]*>").unwrap());
static HTML_TAG: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)<[^>]+>").unwrap());
static MARKDOWN_MARKER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^(?:#{1,6}\s|[-*+]\s|>\s)|\[[^\]]+\]\([^)]+\)|\*\*[^*]+\*\*").unwrap()
});
static MARKDOWN_LINK: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"!?\[([^\]]*)\]\([^)]+\)").unwrap());
static INLINE_CODE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        "{}([^{}]*){}",
        char::from(96),
        char::from(96),
        char::from(96)
    ))
    .unwrap()
});
static STRONG_STAR: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\*\*([^*]+)\*\*").unwrap());
static STRONG_UNDER: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"__([^_]+)__").unwrap());
static EMPHASIS_STAR: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\*([^*]+)\*").unwrap());
static EMPHASIS_UNDER: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"_([^_]+)_").unwrap());
static STRIKE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"~~([^~]+)~~").unwrap());
static SPACE_RUN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[\p{Zs}\t]+").unwrap());
static AROUND_NEWLINE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[ \t]*\n[ \t]*").unwrap());
static NEWLINE_RUN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\n+").unwrap());
static MISSING_PERIOD_SPACE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"([\p{Ll}\p{Nd}]\.["'’”)\]]*)(\p{Lu})"#).unwrap());
static MISSING_PUNCTUATION_SPACE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"([!?…]["'’”)\]]*)(\p{Lu})"#).unwrap());

pub fn clean_text(input: &str, options: TextCleaningOptions) -> CleanedText {
    let sanitized = input.replace("\r\n", "\n").replace('\r', "\n");
    if !options.enabled {
        return CleanedText {
            text: sanitized.trim().to_owned(),
            source_format: "Plain text",
            removed_code_blocks: 0,
            normalized_whitespace: false,
        };
    }
    let mut value = sanitized;
    let mut source_format = "Plain text";
    let mut removed_code_blocks = 0;
    if options.strip_html && HTML_MARKER.is_match(&value) {
        source_format = "HTML";
        value = strip_html(&value);
    } else if MARKDOWN_MARKER.is_match(&value) || contains_fence(&value) {
        source_format = "Markdown";
        if options.strip_code_blocks {
            let result = remove_fenced_code(&value);
            value = result.0;
            removed_code_blocks = result.1;
        }
        if options.strip_markdown {
            value = strip_markdown(&value);
        }
    }
    let before_normalization = value.clone();
    value = normalize(value, options);
    CleanedText {
        normalized_whitespace: value != before_normalization,
        text: value,
        source_format,
        removed_code_blocks,
    }
}

fn strip_html(input: &str) -> String {
    let value = HTML_HIDDEN.replace_all(input, "");
    let value = HTML_BLOCKS.replace_all(&value, "\n");
    let value = HTML_CELLS.replace_all(&value, " ");
    html_escape::decode_html_entities(&HTML_TAG.replace_all(&value, "")).into_owned()
}

fn contains_fence(input: &str) -> bool {
    input.lines().any(|line| {
        let line = line.trim_start();
        line.starts_with(&char::from(96).to_string().repeat(3)) || line.starts_with("~~~")
    })
}

fn remove_fenced_code(input: &str) -> (String, usize) {
    let backtick_fence = char::from(96).to_string().repeat(3);
    let mut output = Vec::new();
    let mut fence: Option<&str> = None;
    let mut removed = 0;
    for line in input.lines() {
        let trimmed = line.trim_start();
        if let Some(active) = fence {
            if trimmed.starts_with(active) {
                fence = None;
            }
            continue;
        }
        let marker = if trimmed.starts_with(&backtick_fence) {
            Some(backtick_fence.as_str())
        } else if trimmed.starts_with("~~~") {
            Some("~~~")
        } else {
            None
        };
        if let Some(marker) = marker {
            fence = Some(marker);
            removed += 1;
        } else {
            output.push(line);
        }
    }
    (output.join("\n"), removed)
}

fn strip_markdown(input: &str) -> String {
    input
        .lines()
        .map(|line| {
            let mut value = line.trim().to_owned();
            if !value.is_empty()
                && value
                    .chars()
                    .all(|value| "-*_=".contains(value) || value.is_whitespace())
            {
                return String::new();
            }
            if let Some(index) = value.find(|value: char| !matches!(value, '#' | '>' | ' ')) {
                value = value[index..].to_owned();
            }
            value = MARKDOWN_LINK.replace_all(&value, "$1").into_owned();
            value = INLINE_CODE.replace_all(&value, "$1").into_owned();
            value = STRONG_STAR.replace_all(&value, "$1").into_owned();
            value = STRONG_UNDER.replace_all(&value, "$1").into_owned();
            value = EMPHASIS_STAR.replace_all(&value, "$1").into_owned();
            value = EMPHASIS_UNDER.replace_all(&value, "$1").into_owned();
            STRIKE.replace_all(&value, "$1").into_owned()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn normalize(input: String, options: TextCleaningOptions) -> String {
    let canonical = input
        .nfc()
        .collect::<String>()
        .replace('\u{2029}', "\n\n")
        .replace(['\u{2028}', '\u{0085}', '\u{000b}'], "\n")
        .replace('\u{000c}', "\n\n");
    let mut value = if options.strip_special_characters {
        canonical
            .chars()
            .filter(|value| !value.is_control() || matches!(value, '\n' | '\t' | ' '))
            .filter(|value| !matches!(value, '\u{00ad}' | '\u{200b}' | '\u{2060}' | '\u{feff}'))
            .map(|value| if value == '\u{fffc}' { ' ' } else { value })
            .collect()
    } else {
        canonical
    };
    if options.normalize_whitespace {
        value = SPACE_RUN.replace_all(&value, " ").into_owned();
        value = AROUND_NEWLINE.replace_all(&value, "\n").into_owned();
        value = NEWLINE_RUN.replace_all(&value, "\n").into_owned();
        value = MISSING_PERIOD_SPACE
            .replace_all(&value, "$1 $2")
            .into_owned();
        value = MISSING_PUNCTUATION_SPACE
            .replace_all(&value, "$1 $2")
            .into_owned();
    }
    value.trim().to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cleans_markdown_code_and_clipboard_artifacts() {
        let fence = char::from(96).to_string().repeat(3);
        let input = format!(
            "# Heading\n\nA **useful** [link](https://example.com).\u{200b}\n{fence}rust\nunsafe {{}}\n{fence}"
        );
        let cleaned = clean_text(&input, TextCleaningOptions::default());
        assert_eq!(cleaned.text, "Heading\nA useful link.");
        assert_eq!(cleaned.source_format, "Markdown");
        assert_eq!(cleaned.removed_code_blocks, 1);
    }

    #[test]
    fn cleans_html_and_preserves_reading_boundaries() {
        let cleaned = clean_text(
            "<article><h1>Title &amp; more</h1><script>ignore()</script><p>First.</p><p>Second.</p></article>",
            TextCleaningOptions::default(),
        );
        assert_eq!(cleaned.text, "Title & more\nFirst.\nSecond.");
        assert_eq!(cleaned.source_format, "HTML");
    }
}
