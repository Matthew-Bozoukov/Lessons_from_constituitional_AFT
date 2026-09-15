import {
  Bot,
  Braces,
  Brain,
  MessageSquare,
  Settings2,
  UserRound,
  Wrench,
} from "lucide-react";
import { DialogueMessage } from "@/lib/content";
import { toolCallView, type ToolSchemaView } from "@/lib/records";

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
  tools = [],
}: {
  messages: DialogueMessage[];
  compact?: boolean;
  /** The row's declared tool schemas, shown once above the turns, family-agnostic. */
  tools?: ToolSchemaView[];
}) {
  return (
    <div className={compact ? "dialogue compact" : "dialogue"}>
      {tools.length > 0 && (
        <details className="dialogue-think dialogue-tools dialogue-tool-schemas">
          <summary><Wrench size={13} /> Tools available ({tools.length})</summary>
          <div className="dialogue-think-content">
            {tools.map((tool) => (
              <pre className="dialogue-tool-call" key={tool.name}>
                <b>{tool.name}</b>
                {tool.parameters.length > 0 ? `(${tool.parameters.join(", ")})` : "()"}
                {tool.description ? `\n${tool.description}` : ""}
              </pre>
            ))}
          </div>
        </details>
      )}
      {messages.map((message, index) => {
        const config = roleConfig[message.role] || {
          label: message.role || "Message",
          icon: MessageSquare,
          className: "unknown",
        };
        const Icon = config.icon;
        const reasoning =
          typeof message.reasoning_content === "string"
            ? message.reasoning_content.trim()
            : "";
        const toolCalls = Array.isArray(message.tool_calls)
          ? (message.tool_calls as unknown[]).map(toolCallView)
          : [];
        return (
          <article className={`dialogue-turn ${config.className}`} key={`${message.role}-${index}`}>
            <header>
              <span><Icon size={14} /> {config.label}</span>
              <code>{String(index + 1).padStart(2, "0")}</code>
            </header>
            {reasoning && (
              <details className="dialogue-think">
                <summary><Brain size={13} /> Reasoning trace</summary>
                <div className="dialogue-think-content">{reasoning}</div>
              </details>
            )}
            {toolCalls.length > 0 && (
              <details className="dialogue-think dialogue-tools">
                <summary><Wrench size={13} /> Tool calls ({toolCalls.length})</summary>
                <div className="dialogue-think-content">
                  {toolCalls.map((call, i) => (
                    <pre className="dialogue-tool-call" key={i}>
                      <b>{call.name}</b>{"\n"}{call.arguments}
                    </pre>
                  ))}
                </div>
              </details>
            )}
            {message.content && <div className="dialogue-content">{message.content}</div>}
          </article>
        );
      })}
    </div>
  );
}

