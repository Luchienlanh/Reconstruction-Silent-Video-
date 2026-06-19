import { BookOpen, ExternalLink, FileText, Gavel } from "lucide-react";
import type { Citation, SourceType } from "../types/legal";
import { sourceTypeLabels } from "../types/legal";

interface CitationListProps {
  citations: Citation[];
  onCitationSelect: (citation: Citation) => void;
  selectedCitationId?: string;
}

const renderSourceIcon = (sourceType: SourceType) => {
  if (sourceType === "phap-dien") {
    return <BookOpen aria-hidden="true" size={15} />;
  }

  if (sourceType === "an-le") {
    return <Gavel aria-hidden="true" size={15} />;
  }

  return <FileText aria-hidden="true" size={15} />;
};

export function CitationList({
  citations,
  onCitationSelect,
  selectedCitationId,
}: CitationListProps) {
  if (citations.length === 0) {
    return null;
  }

  return (
    <section aria-label="Nguồn trích dẫn" className="citation-list">
      <div className="citation-list__heading">Nguồn tham chiếu</div>
      <div className="citation-list__items">
        {citations.map((citation) => (
          <article
            className={`citation-card source-${citation.sourceType} ${
              selectedCitationId === citation.id ? "is-selected" : ""
            }`}
            key={citation.id}
          >
            <div className="citation-card__topline">
              <span className="source-badge">
                {renderSourceIcon(citation.sourceType)}
                {sourceTypeLabels[citation.sourceType]}
              </span>
              {citation.status ? (
                <span className="status-chip">{citation.status}</span>
              ) : null}
            </div>
            <h3>{citation.title}</h3>
            <dl className="citation-card__meta">
              {citation.article ? (
                <div>
                  <dt>Điều khoản</dt>
                  <dd>
                    {citation.article}
                    {citation.clause ? `, ${citation.clause}` : ""}
                  </dd>
                </div>
              ) : null}
              {citation.effectiveDate ? (
                <div>
                  <dt>Hiệu lực</dt>
                  <dd>{citation.effectiveDate}</dd>
                </div>
              ) : null}
            </dl>
            <p>{citation.summary}</p>
            <button
              className="citation-card__action"
              onClick={() => onCitationSelect(citation)}
              type="button"
            >
              <ExternalLink aria-hidden="true" size={15} />
              Xem chi tiết
            </button>
          </article>
        ))}
      </div>
    </section>
  );
}
