import { useEffect, useMemo, useState } from "react";

export type ThemePreference = "light" | "dark" | "system";
export type ResolvedTheme = "light" | "dark";

const STORAGE_KEY = "legal-chat-theme";

const getStoredPreference = (): ThemePreference => {
  if (typeof window === "undefined") {
    return "system";
  }

  const stored = window.localStorage.getItem(STORAGE_KEY);
  return stored === "light" || stored === "dark" || stored === "system"
    ? stored
    : "system";
};

const getSystemTheme = (): ResolvedTheme => {
  if (typeof window === "undefined") {
    return "light";
  }

  return window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
};

export function useTheme() {
  const [preference, setPreferenceState] =
    useState<ThemePreference>(getStoredPreference);
  const [systemTheme, setSystemTheme] = useState<ResolvedTheme>(getSystemTheme);

  const resolvedTheme = useMemo<ResolvedTheme>(
    () => (preference === "system" ? systemTheme : preference),
    [preference, systemTheme]
  );

  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const updateSystemTheme = () => {
      setSystemTheme(media.matches ? "dark" : "light");
    };

    updateSystemTheme();
    media.addEventListener("change", updateSystemTheme);

    return () => {
      media.removeEventListener("change", updateSystemTheme);
    };
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = resolvedTheme;
    document.documentElement.dataset.themePreference = preference;
    document.documentElement.style.colorScheme = resolvedTheme;
  }, [preference, resolvedTheme]);

  const setPreference = (nextPreference: ThemePreference) => {
    window.localStorage.setItem(STORAGE_KEY, nextPreference);
    setPreferenceState(nextPreference);
  };

  return {
    preference,
    resolvedTheme,
    setPreference,
  };
}
