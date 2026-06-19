import { AlertTriangle, LoaderCircle } from "lucide-react";
import type { ChatMessage, Citation } from "../types/legal";
import { CitationList } from "./CitationList";

interface MessageBubbleProps {
  message: ChatMessage;
  onCitationSelect: (citation: Citation) => void;
  selectedCitationId?: string;
}

const splitParagraphs = (content: string) =>
  content.split("\n").filter((paragraph) => paragraph.trim().length > 0);

export function MessageBubble({
  message,
  onCitationSelect,
  selectedCitationId,
}: MessageBubbleProps) {
  const isAssistant = message.role === "assistant";

  return (
    <article
      aria-live={message.status === "streaming" ? "polite" : undefined}
      className={`message-row ${message.role}`}
    >
      <div className="message-avatar" aria-hidden="true">
        {isAssistant ? "PL" : "Bạn"}
      </div>
      <div className="message-content">
        {message.status === "loading" ? (
          <div className="answer-loading">
            <div className="loading-line wide" />
            <div className="loading-line" />
            <div className="loading-line short" />
          </div>
        ) : null}

        {message.status === "error" ? (
          <div className="state-card error-state">
            <AlertTriangle aria-hidden="true" size={18} />
            <div>
              <strong>Không thể hoàn tất tra cứu</strong>
              <p>{message.content}</p>
            </div>
          </div>
        ) : null}

        {message.status === "no-result" ? (
          <div className="state-card no-result-state">
            <AlertTriangle aria-hidden="true" size={18} />
            <div>
              <strong>Không tìm thấy nguồn phù hợp</strong>
              <p>{message.content}</p>
            </div>
          </div>
        ) : null}

        {message.status !== "loading" &&
        message.status !== "error" &&
        message.status !== "no-result" ? (
          <div className="answer-text">
            {splitParagraphs(message.content).map((paragraph) => (
              <p key={paragraph}>{paragraph}</p>
            ))}
            {message.status === "streaming" ? (
              <span className="streaming-status">
                <LoaderCircle aria-hidden="true" size={14} />
                Đang soạn câu trả lời
              </span>
            ) : null}
          </div>
        ) : null}

        {isAssistant &&
        message.status !== "loading" &&
        message.status !== "error" &&
        message.citations?.length ? (
          <CitationList
            citations={message.citations}
            onCitationSelect={onCitationSelect}
            selectedCitationId={selectedCitationId}
          />
        ) : null}

        {isAssistant && message.followUps?.length ? (
          <div className="follow-ups" aria-label="Câu hỏi tiếp theo">
            {message.followUps.map((item) => (
              <span key={item}>{item}</span>
            ))}
          </div>
        ) : null}
      </div>
    </article>
  );
}
