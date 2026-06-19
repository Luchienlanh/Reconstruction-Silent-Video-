import { FormEvent, useEffect, useRef, useState } from "react";
import { Send } from "lucide-react";
import type { Citation, Conversation } from "../types/legal";
import { MessageBubble } from "./MessageBubble";
import { PromptSuggestions } from "./PromptSuggestions";

interface ChatThreadProps {
  conversation: Conversation;
  prompts: string[];
  isSubmitting: boolean;
  selectedCitationId?: string;
  onSend: (question: string) => void;
  onCitationSelect: (citation: Citation) => void;
}

export function ChatThread({
  conversation,
  prompts,
  isSubmitting,
  selectedCitationId,
  onSend,
  onCitationSelect,
}: ChatThreadProps) {
  const [draft, setDraft] = useState("");
  const scrollAnchorRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollAnchorRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [conversation.messages]);

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const question = draft.trim();

    if (!question || isSubmitting) {
      return;
    }

    setDraft("");
    onSend(question);
  };

  const sendPrompt = (prompt: string) => {
    if (isSubmitting) {
      return;
    }

    setDraft("");
    onSend(prompt);
  };

  const hasMessages = conversation.messages.length > 0;

  return (
    <section className="chat-thread" aria-label="Không gian trò chuyện">
      <header className="thread-header">
        <div>
          <span className="thread-header__category">{conversation.category}</span>
          <h2>{conversation.title}</h2>
        </div>
        <div className="thread-header__status">
          <span className="status-dot" />
          Dữ liệu mẫu có cấu trúc
        </div>
      </header>

      <div className="messages-viewport">
        {hasMessages ? (
          conversation.messages.map((message) => (
            <MessageBubble
              key={message.id}
              message={message}
              onCitationSelect={onCitationSelect}
              selectedCitationId={selectedCitationId}
            />
          ))
        ) : (
          <div className="empty-thread">
            <span className="empty-thread__label">Phiên tra cứu mới</span>
            <h3>Bắt đầu bằng một câu hỏi pháp lý cụ thể.</h3>
            <p>
              Nên nêu lĩnh vực, mốc thời gian hoặc loại nguồn cần kiểm tra để
              câu trả lời bám sát căn cứ hơn.
            </p>
            <PromptSuggestions prompts={prompts.slice(0, 3)} onSelect={sendPrompt} />
          </div>
        )}
        <div ref={scrollAnchorRef} />
      </div>

      {hasMessages ? (
        <PromptSuggestions compact prompts={prompts.slice(0, 3)} onSelect={sendPrompt} />
      ) : null}

      <form className="composer" onSubmit={submit}>
        <label className="sr-only" htmlFor="legal-question">
          Nhập câu hỏi pháp lý
        </label>
        <textarea
          id="legal-question"
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              event.currentTarget.form?.requestSubmit();
            }
          }}
          placeholder="Nhập câu hỏi về văn bản pháp luật, pháp điển hoặc án lệ..."
          rows={2}
          value={draft}
        />
        <button disabled={!draft.trim() || isSubmitting} type="submit">
          <Send aria-hidden="true" size={17} />
          Gửi
        </button>
      </form>
      <p className="disclaimer">
        Thông tin chỉ hỗ trợ tra cứu, không thay thế tư vấn pháp lý.
      </p>
    </section>
  );
}
