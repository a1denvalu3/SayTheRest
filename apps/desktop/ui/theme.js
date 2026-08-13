const themePreferenceKey = 'color-theme';
const systemTheme = window.matchMedia('(prefers-color-scheme: dark)');

function savedThemePreference() {
  const preference = localStorage.getItem(themePreferenceKey);
  return ['light', 'dark'].includes(preference) ? preference : 'system';
}

function applyTheme(preference = savedThemePreference()) {
  const resolved = preference === 'system' ? (systemTheme.matches ? 'dark' : 'light') : preference;
  document.documentElement.dataset.theme = resolved;
  document.documentElement.style.colorScheme = resolved;
}

applyTheme();
systemTheme.addEventListener('change', () => {
  if (savedThemePreference() === 'system') applyTheme('system');
});

document.addEventListener('DOMContentLoaded', () => {
  const preference = savedThemePreference();
  const selected = document.querySelector(`[name="color_theme"][value="${preference}"]`);
  if (selected) selected.checked = true;
  document.querySelectorAll('[name="color_theme"]').forEach(input => input.addEventListener('change', event => {
    const value = event.currentTarget.value;
    if (value === 'system') localStorage.removeItem(themePreferenceKey);
    else localStorage.setItem(themePreferenceKey, value);
    applyTheme(value);
  }));
});
