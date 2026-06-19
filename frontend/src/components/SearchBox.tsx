import { Search } from "lucide-react";

interface SearchBoxProps {
  value: string;
  onChange: (value: string) => void;
  placeholder: string;
  label: string;
  className?: string;
}

export function SearchBox({
  value,
  onChange,
  placeholder,
  label,
  className,
}: SearchBoxProps) {
  return (
    <label className={`search-box ${className ?? ""}`}>
      <span className="sr-only">{label}</span>
      <Search aria-hidden="true" size={16} />
      <input
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder={placeholder}
        type="search"
      />
    </label>
  );
}
