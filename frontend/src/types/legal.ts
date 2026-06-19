export type SourceType = "van-ban" | "phap-dien" | "an-le";

export type MessageRole = "user" | "assistant";

export type MessageStatus =
  | "complete"
  | "loading"
  | "streaming"
  | "error"
  | "no-result";

export type ConversationCategory =
  | "Tất cả"
  | "Lao động"
  | "Đất đai"
  | "Dân sự"
  | "Hành chính"
  | "Hình sự"
  | "Án lệ";

export type ConfidenceLevel = "cao" | "trung bình" | "thấp";

export interface Citation {
  id: string;
  sourceType: SourceType;
  title: string;
  article?: string;
  clause?: string;
  effectiveDate?: string;
  status?: string;
  agency?: string;
  detailUrl?: string;
  summary: string;
  excerpt: string;
  related?: string[];
}

export interface LegalAnswer {
  answer: string;
  citations: Citation[];
  confidence: ConfidenceLevel;
  followUps: string[];
  noResultReason?: string;
}

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  status: MessageStatus;
  createdAt: string;
  citations?: Citation[];
  confidence?: ConfidenceLevel;
  followUps?: string[];
}

export interface Conversation {
  id: string;
  title: string;
  category: Exclude<ConversationCategory, "Tất cả">;
  updatedAt: string;
  lastSummary: string;
  messages: ChatMessage[];
}

export interface ChatRequest {
  question: string;
  topic?: ConversationCategory;
  conversationId?: string;
}

export const sourceTypeLabels: Record<SourceType, string> = {
  "van-ban": "Văn bản pháp luật",
  "phap-dien": "Pháp điển",
  "an-le": "Án lệ",
};
