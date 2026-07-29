import type { Metadata } from "next";
import "katex/dist/katex.min.css";
import "./globals.css";
import { SiteHeader } from "./components/SiteHeader";

export const metadata: Metadata = {
  title: {
    default: "Synthetic Finetuning for Constitution: Research Log",
    template: "%s · SFC Research Log",
  },
  description:
    "Local research log for synthetic constitution-oriented finetuning, evaluation, and generalization work.",
  icons: {
    icon: "/favicon.png",
    shortcut: "/favicon.png",
  },
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <SiteHeader />
        {children}
        <footer className="site-footer">
          <div className="page-container">
            <span>SFC Research Log</span>
            <span>Immutable sources · generated index · local only</span>
          </div>
        </footer>
      </body>
    </html>
  );
}
