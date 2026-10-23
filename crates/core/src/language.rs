//! Language identifiers shared by Qwen configuration and user-facing controls.
use anyhow::{Context, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QwenLanguage {
    pub code: &'static str,
    pub name: &'static str,
}

pub const QWEN_LANGUAGES: &[QwenLanguage] = &[
    QwenLanguage {
        code: "auto",
        name: "Auto",
    },
    QwenLanguage {
        code: "zh",
        name: "Chinese",
    },
    QwenLanguage {
        code: "en",
        name: "English",
    },
    QwenLanguage {
        code: "ja",
        name: "Japanese",
    },
    QwenLanguage {
        code: "ko",
        name: "Korean",
    },
    QwenLanguage {
        code: "de",
        name: "German",
    },
    QwenLanguage {
        code: "fr",
        name: "French",
    },
    QwenLanguage {
        code: "ru",
        name: "Russian",
    },
    QwenLanguage {
        code: "pt",
        name: "Portuguese",
    },
    QwenLanguage {
        code: "es",
        name: "Spanish",
    },
    QwenLanguage {
        code: "it",
        name: "Italian",
    },
];

/// Accept existing backend names as well as catalog codes. Unknown values must
/// never silently select automatic language detection in the native worker.
pub fn qwen_language(value: &str) -> Result<&'static QwenLanguage> {
    let value = value.trim();
    let value = if value.eq_ignore_ascii_case("cmn") {
        "zh"
    } else {
        value
    };
    QWEN_LANGUAGES
        .iter()
        .find(|language| {
            language.code.eq_ignore_ascii_case(value) || language.name.eq_ignore_ascii_case(value)
        })
        .context(
            "unsupported Qwen language; choose auto, zh, en, ja, ko, de, fr, ru, pt, es, or it",
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_and_existing_names_resolve_to_the_same_backend_language() {
        for language in QWEN_LANGUAGES {
            assert_eq!(qwen_language(language.code).unwrap(), language);
            assert_eq!(
                qwen_language(&language.name.to_lowercase()).unwrap(),
                language
            );
        }
        assert_eq!(qwen_language("cmn").unwrap().code, "zh");
        for value in [
            "",
            "automatic",
            "es-invalid",
            "Arabic",
            "fr\0",
            "Japanese\nEnglish",
        ] {
            assert!(qwen_language(value).is_err(), "must reject {value:?}");
        }
    }
}
