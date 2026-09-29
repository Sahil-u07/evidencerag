import { useRef, useState } from "react";
import type { DocumentItem } from "../types";

type DocumentSidebarProps = {
  documents: DocumentItem[];
  open: boolean;
  onClose: () => void;
  onUpload: (file: File) => Promise<void>;
  onDelete: (name: string) => Promise<void>;
  uploading: boolean;
  error: string;
};

export default function DocumentSidebar({
  documents,
  open,
  onClose,
  onUpload,
  onDelete,
  uploading,
  error,
}: DocumentSidebarProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(
    null,
  );

  const [search, setSearch] = useState("");

  const filteredDocuments = documents.filter((document) =>
    document.name
      .toLowerCase()
      .includes(search.toLowerCase()),
  );

  const handleFileChange = async (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.[0];

    if (!file) {
      return;
    }

    await onUpload(file);

    event.target.value = "";
  };

  return (
    <>
      {open && (
        <button
          type="button"
          className="sidebar-backdrop"
          aria-label="Close documents panel"
          onClick={onClose}
        />
      )}

      <aside
        className={`document-sidebar ${
          open ? "sidebar-open" : ""
        }`}
        aria-label="Documents"
      >
        <div className="sidebar-header">
          <div>
            <span className="sidebar-eyebrow">
              Knowledge base
            </span>

            <h2>Your documents</h2>

            <p>
              Upload files and ask questions about
              them.
            </p>
          </div>

          <button
            type="button"
            className="sidebar-close"
            onClick={onClose}
            aria-label="Close documents"
          >
            ×
          </button>
        </div>

        <div className="sidebar-upload">
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.txt,.md"
            onChange={handleFileChange}
            hidden
          />

          <button
            type="button"
            className="upload-button"
            onClick={() =>
              fileInputRef.current?.click()
            }
            disabled={uploading}
          >
            <span aria-hidden="true">
              {uploading ? "…" : "+"}
            </span>

            {uploading
              ? "Indexing document…"
              : "Add document"}
          </button>

          <p>
            PDF, TXT, or Markdown · up to 10 MB
          </p>
        </div>

        {error && (
          <div
            className="sidebar-error"
            role="alert"
          >
            {error}
          </div>
        )}

        <div className="document-search">
          <label htmlFor="document-filter">
            Filter documents
          </label>

          <input
            id="document-filter"
            type="search"
            placeholder="Search files…"
            value={search}
            onChange={(event) =>
              setSearch(event.target.value)
            }
          />
        </div>

        <div className="document-count">
          <span>Documents</span>

          <strong>
            {filteredDocuments.length}
          </strong>
        </div>

        <div className="document-list">
          {filteredDocuments.length === 0 ? (
            <div className="empty-documents">
              <div
                className="empty-documents-icon"
                aria-hidden="true"
              >
                ◫
              </div>

              <h3>
                {documents.length === 0
                  ? "No documents yet"
                  : "No matching documents"}
              </h3>

              <p>
                {documents.length === 0
                  ? "Add a PDF, text, or Markdown file to start asking questions."
                  : "Try a different file name."}
              </p>
            </div>
          ) : (
            filteredDocuments.map((document) => (
              <article
                className="document-item"
                key={document.name}
              >
                <div className="document-icon">
                  {getFileIcon(document.name)}
                </div>

                <div className="document-info">
                  <strong title={document.name}>
                    {document.name}
                  </strong>

                  <span>
                    {document.status ||
                      "Indexed"}
                  </span>
                </div>

                <button
                  type="button"
                  className="document-delete"
                  onClick={() =>
                    onDelete(document.name)
                  }
                  aria-label={`Delete ${document.name}`}
                  title="Delete document"
                >
                  ×
                </button>
              </article>
            ))
          )}
        </div>

        <div className="sidebar-footer">
          <span className="status-dot" />

          <span>
            {documents.length > 0
              ? "Ready to search"
              : "Waiting for documents"}
          </span>
        </div>
      </aside>
    </>
  );
}

function getFileIcon(name: string) {
  const extension = name
    .split(".")
    .pop()
    ?.toLowerCase();

  if (extension === "pdf") {
    return "PDF";
  }

  if (extension === "md") {
    return "MD";
  }

  return "TXT";
}