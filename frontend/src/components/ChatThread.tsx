import { useEffect } from "react";
import type { ChatMessage } from "../types";
import MessageBubble from "./MessageBubble";

type Props = {
  messages: ChatMessage[];
  onRetry: (query: string) => void;
  onDemo: () => void;
};

export default function ChatThread({
  messages,
  onRetry,
  onDemo,
}: Props) {
  useEffect(() => {
    window.scrollTo({
      top: document.documentElement.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  return (
    <section
      className="chat-thread"
      aria-label="Conversation"
    >
      <div className="thread-inner">
        {messages.map((message) => (
          <MessageBubble
            key={message.id}
            message={message}
            onRetry={onRetry}
            onDemo={onDemo}
          />
        ))}
      </div>
    </section>
  );
}