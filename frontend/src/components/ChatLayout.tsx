import type { ReactNode } from "react";
import { Menu, PanelRightOpen, X } from "lucide-react";

interface ChatLayoutProps {
  sidebar: ReactNode;
  thread: ReactNode;
  sourcePanel: ReactNode;
  isSidebarOpen: boolean;
  isSourceOpen: boolean;
  hasSelectedSource: boolean;
  onOpenSidebar: () => void;
  onCloseSidebar: () => void;
  onOpenSource: () => void;
  onCloseSource: () => void;
}

export function ChatLayout({
  sidebar,
  thread,
  sourcePanel,
  isSidebarOpen,
  isSourceOpen,
  hasSelectedSource,
  onOpenSidebar,
  onCloseSidebar,
  onOpenSource,
  onCloseSource,
}: ChatLayoutProps) {
  return (
    <div className="chat-layout">
      <header className="mobile-topbar">
        <button
          aria-label="Mở lịch sử hội thoại"
          className="icon-button"
          onClick={onOpenSidebar}
          type="button"
        >
          <Menu aria-hidden="true" size={20} />
        </button>
        <div className="mobile-brand">
          <span>Tra cứu pháp luật</span>
          <small>Văn bản, pháp điển, án lệ</small>
        </div>
        <button
          aria-label="Mở chi tiết nguồn"
          className="icon-button"
          disabled={!hasSelectedSource}
          onClick={onOpenSource}
          type="button"
        >
          <PanelRightOpen aria-hidden="true" size={20} />
        </button>
      </header>

      <aside className={`sidebar-shell ${isSidebarOpen ? "is-open" : ""}`}>
        <div className="drawer-header">
          <span>Lịch sử</span>
          <button
            aria-label="Đóng lịch sử hội thoại"
            className="icon-button"
            onClick={onCloseSidebar}
            type="button"
          >
            <X aria-hidden="true" size={18} />
          </button>
        </div>
        {sidebar}
      </aside>

      <main className="main-shell">{thread}</main>

      <aside className={`source-shell ${isSourceOpen ? "is-open" : ""}`}>
        {sourcePanel}
      </aside>

      <button
        aria-label="Đóng lớp phủ"
        className={`drawer-scrim ${
          isSidebarOpen || isSourceOpen ? "is-visible" : ""
        }`}
        onClick={() => {
          onCloseSidebar();
          onCloseSource();
        }}
        type="button"
      />
    </div>
  );
}
