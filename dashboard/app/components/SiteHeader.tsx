"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Activity,
  Beaker,
  BrainCircuit,
  Database,
  FlaskConical,
  Lightbulb,
  ListOrdered,
  Menu,
  ScanSearch,
  X,
} from "lucide-react";
import { useState } from "react";

const nav = [
  { href: "/", label: "Overview", icon: Activity },
  { href: "/logs", label: "Logs", icon: FlaskConical },
  { href: "/evals", label: "Evals", icon: Beaker },
  { href: "/datasets", label: "Datasets", icon: Database },
  { href: "/selection", label: "Selection", icon: ListOrdered },
  { href: "/petri", label: "Petri", icon: ScanSearch },
  { href: "/models", label: "Models", icon: BrainCircuit },
  { href: "/findings", label: "Findings", icon: Lightbulb },
];

export function SiteHeader() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  return (
    <header className="site-header">
      <div className="header-inner">
        <Link href="/" className="brand" onClick={() => setOpen(false)}>
          <span className="brand-mark" aria-hidden="true">
            <span />
            <span />
            <span />
          </span>
          <span className="brand-copy">
            <span className="brand-kicker">SFC</span>
            <span className="brand-name">Research Log</span>
          </span>
        </Link>

        <button
          className="mobile-menu"
          type="button"
          aria-label={open ? "Close navigation" : "Open navigation"}
          aria-expanded={open}
          onClick={() => setOpen((value) => !value)}
        >
          {open ? <X size={19} /> : <Menu size={19} />}
        </button>

        <nav className={open ? "primary-nav is-open" : "primary-nav"} aria-label="Main navigation">
          {nav.map(({ href, label, icon: Icon }) => {
            const active =
              href === "/" ? pathname === "/" : pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                className={active ? "nav-link is-active" : "nav-link"}
                onClick={() => setOpen(false)}
              >
                <Icon size={15} strokeWidth={1.8} />
                {label}
              </Link>
            );
          })}
        </nav>

        <div className="local-indicator" title="This application is local only">
          <span />
          Local corpus
        </div>
      </div>
    </header>
  );
}
