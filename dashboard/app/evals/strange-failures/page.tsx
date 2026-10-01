// ABOUTME: Findings from evaluation failures, linked to live HF runs and original evidence.
// ABOUTME: SWE-bench loops are browsed by official outcome and exact repeated transcript spans.
import type { Metadata } from 'next';
import Link from 'next/link';
import { SweBenchDiscovery } from '../../components/SweBenchBrowser';
export const metadata: Metadata = { title: 'Strange failure findings · Evals' };
export default function StrangeFailuresPage() {
  return <main className="page-container inner-page"><header className="page-heading"><div>
    <span className="eyebrow"><Link href="/evals">Evaluations</Link> / Failure investigations</span>
    <h1>Strange failure findings</h1>
    <p>A model can spend its response repeating the same passage instead of taking its next action. Inspect the original text, compare consecutive copies, and browse both successful and failed tasks from the same run.</p>
  </div></header><SweBenchDiscovery /></main>;
}
