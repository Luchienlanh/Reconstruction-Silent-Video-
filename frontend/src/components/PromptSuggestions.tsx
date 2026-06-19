import { Scale } from "lucide-react";

interface PromptSuggestionsProps {
  prompts: string[];
  onSelect: (prompt: string) => void;
  compact?: boolean;
}

export function PromptSuggestions({
  prompts,
  onSelect,
  compact = false,
}: PromptSuggestionsProps) {
  return (
    <div className={compact ? "prompt-suggestions compact" : "prompt-suggestions"}>
      <div className="prompt-suggestions__header">
        <Scale aria-hidden="true" size={16} />
        <span>Gợi ý tra cứu</span>
      </div>
      <div className="prompt-suggestions__grid">
        {prompts.map((prompt) => (
          <button
            className="prompt-button"
            key={prompt}
            onClick={() => onSelect(prompt)}
            type="button"
          >
            {prompt}
          </button>
        ))}
      </div>
    </div>
  );
}
