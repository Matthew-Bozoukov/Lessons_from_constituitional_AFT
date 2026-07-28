import {
  Bot,
  Braces,
  MessageSquare,
  Settings2,
  UserRound,
  Wrench,
} from "lucide-react";
import { DialogueMessage } from "@/lib/content";

const roleConfig: Record<
  string,
  { label: string; icon: typeof UserRound; className: string }
> = {
  system: { label: "System", icon: Settings2, className: "system" },
  user: { label: "User", icon: UserRound, className: "user" },
  assistant: { label: "Assistant", icon: Bot, className: "assistant" },
  tool: { label: "Tool", icon: Wrench, className: "tool" },
  developer: { label: "Developer", icon: Braces, className: "system" },
};

export function DialogueTranscript({
  messages,
  compact = false,
}: {
  messages: DialogueMessage[];
  compact?: boolean;
}) {
  return (
    <div className={compact ? "dialogue compact" : "dialogue"}>
      {messages.map((message, index) => {
        const config = roleConfig[message.role] || {
          label: message.role || "Message",
          icon: MessageSquare,
          className: "unknown",
        };
        const Icon = config.icon;
        return (
          <article className={`dialogue-turn ${config.className}`} key={`${message.role}-${index}`}>
            <header>
              <span><Icon size={14} /> {config.label}</span>
              <code>{String(index + 1).padStart(2, "0")}</code>
            </header>
            <div className="dialogue-content">{message.content}</div>
          </article>
        );
      })}
    </div>
  );
}

