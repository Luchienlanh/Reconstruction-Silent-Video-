import {
  MessageSquare,
  Monitor,
  Moon,
  Plus,
  Scale,
  SlidersHorizontal,
  Sun,
} from "lucide-react";
import type { ThemePreference } from "../hooks/useTheme";
import type { Conversation, ConversationCategory } from "../types/legal";
import { SearchBox } from "./SearchBox";

interface SidebarProps {
  conversations: Conversation[];
  activeConversationId: string;
  searchValue: string;
  selectedCategory: ConversationCategory;
  themePreference: ThemePreference;
  onSearchChange: (value: string) => void;
  onCategoryChange: (category: ConversationCategory) => void;
  onConversationSelect: (conversationId: string) => void;
  onNewChat: () => void;
  onThemeChange: (theme: ThemePreference) => void;
}

const categories: ConversationCategory[] = [
  "Tất cả",
  "Lao động",
  "Đất đai",
  "Dân sự",
  "Hành chính",
  "Hình sự",
  "Án lệ",
];

const formatDate = (isoDate: string) =>
  new Intl.DateTimeFormat("vi-VN", {
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(isoDate));

export function Sidebar({
  conversations,
  activeConversationId,
  searchValue,
  selectedCategory,
  themePreference,
  onSearchChange,
  onCategoryChange,
  onConversationSelect,
  onNewChat,
  onThemeChange,
}: SidebarProps) {
  return (
    <div className="sidebar">
      <div className="sidebar__brand">
        <div className="brand-mark">
          <Scale aria-hidden="true" size={20} />
        </div>
        <div>
          <h1>Trợ lý pháp luật</h1>
          <p>Tra cứu văn bản, pháp điển, án lệ</p>
        </div>
      </div>

      <button className="new-chat-button" onClick={onNewChat} type="button">
        <Plus aria-hidden="true" size={17} />
        Hội thoại mới
      </button>

      <SearchBox
        label="Tìm trong lịch sử hội thoại"
        onChange={onSearchChange}
        placeholder="Tìm hội thoại, chủ đề..."
        value={searchValue}
      />

      <section className="sidebar-section" aria-label="Bộ lọc chủ đề">
        <div className="sidebar-section__title">
          <SlidersHorizontal aria-hidden="true" size={15} />
          Chủ đề
        </div>
        <div className="category-filter">
          {categories.map((category) => (
            <button
              aria-pressed={selectedCategory === category}
              className={selectedCategory === category ? "is-active" : ""}
              key={category}
              onClick={() => onCategoryChange(category)}
              type="button"
            >
              {category}
            </button>
          ))}
        </div>
      </section>

      <section className="conversation-list" aria-label="Lịch sử hội thoại">
        {conversations.length ? (
          conversations.map((conversation) => (
            <button
              className={`conversation-item ${
                activeConversationId === conversation.id ? "is-active" : ""
              }`}
              key={conversation.id}
              onClick={() => onConversationSelect(conversation.id)}
              type="button"
            >
              <span className="conversation-item__icon">
                <MessageSquare aria-hidden="true" size={16} />
              </span>
              <span className="conversation-item__body">
                <strong>{conversation.title}</strong>
                <span>{conversation.lastSummary}</span>
                <small>
                  {conversation.category} · {formatDate(conversation.updatedAt)}
                </small>
              </span>
            </button>
          ))
        ) : (
          <div className="sidebar-empty">
            Không có hội thoại phù hợp với bộ lọc hiện tại.
          </div>
        )}
      </section>

      <div className="sidebar__footer">
        <div className="theme-switch" aria-label="Chọn giao diện">
          <button
            aria-pressed={themePreference === "light"}
            onClick={() => onThemeChange("light")}
            type="button"
          >
            <Sun aria-hidden="true" size={15} />
            Sáng
          </button>
          <button
            aria-pressed={themePreference === "dark"}
            onClick={() => onThemeChange("dark")}
            type="button"
          >
            <Moon aria-hidden="true" size={15} />
            Tối
          </button>
          <button
            aria-pressed={themePreference === "system"}
            onClick={() => onThemeChange("system")}
            type="button"
          >
            <Monitor aria-hidden="true" size={15} />
            Hệ thống
          </button>
        </div>
      </div>
    </div>
  );
}
