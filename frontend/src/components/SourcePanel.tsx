import { BookOpen, ExternalLink, FileText, Gavel, X } from "lucide-react";
import type { Citation, SourceType } from "../types/legal";
import { sourceTypeLabels } from "../types/legal";

interface SourcePanelProps {
  citation?: Citation;
  onClose: () => void;
}

const renderSourceIcon = (sourceType: SourceType) => {
  if (sourceType === "phap-dien") {
    return <BookOpen aria-hidden="true" size={17} />;
  }

  if (sourceType === "an-le") {
    return <Gavel aria-hidden="true" size={17} />;
  }

  return <FileText aria-hidden="true" size={17} />;
};

export function SourcePanel({ citation, onClose }: SourcePanelProps) {
  return (
    <section aria-label="Chi tiết nguồn" className="source-panel">
      <div className="source-panel__header">
        <div>
          <span className="source-panel__eyebrow">Chi tiết nguồn</span>
          <h2>{citation ? citation.title : "Chưa chọn nguồn"}</h2>
        </div>
        <button
          aria-label="Đóng chi tiết nguồn"
          className="icon-button source-panel__close"
          onClick={onClose}
          type="button"
        >
          <X aria-hidden="true" size={18} />
        </button>
      </div>

      {citation ? (
        <div className="source-panel__body">
          <span className={`source-badge large source-${citation.sourceType}`}>
            {renderSourceIcon(citation.sourceType)}
            {sourceTypeLabels[citation.sourceType]}
          </span>

          <dl className="source-panel__facts">
            {citation.article ? (
              <div>
                <dt>Điều, mục</dt>
                <dd>
                  {citation.article}
                  {citation.clause ? `, ${citation.clause}` : ""}
                </dd>
              </div>
            ) : null}
            {citation.effectiveDate ? (
              <div>
                <dt>Ngày hiệu lực</dt>
                <dd>{citation.effectiveDate}</dd>
              </div>
            ) : null}
            {citation.status ? (
              <div>
                <dt>Trạng thái</dt>
                <dd>{citation.status}</dd>
              </div>
            ) : null}
            {citation.agency ? (
              <div>
                <dt>Cơ quan</dt>
                <dd>{citation.agency}</dd>
              </div>
            ) : null}
          </dl>

          <div className="source-panel__section">
            <h3>Tóm tắt</h3>
            <p>{citation.summary}</p>
          </div>

          <div className="source-panel__section">
            <h3>Đoạn liên quan</h3>
            <blockquote>{citation.excerpt}</blockquote>
          </div>

          {citation.related?.length ? (
            <div className="source-panel__section">
              <h3>Liên quan</h3>
              <ul className="related-list">
                {citation.related.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>
          ) : null}

          {citation.detailUrl ? (
            <a
              className="source-panel__link"
              href={citation.detailUrl}
              rel="noreferrer"
              target="_blank"
            >
              <ExternalLink aria-hidden="true" size={16} />
              Mở nguồn chính thức
            </a>
          ) : null}
        </div>
      ) : (
        <div className="source-panel__empty">
          <FileText aria-hidden="true" size={28} />
          <p>Chọn một trích dẫn trong câu trả lời để xem điều khoản, hiệu lực và đoạn liên quan.</p>
        </div>
      )}
    </section>
  );
}
