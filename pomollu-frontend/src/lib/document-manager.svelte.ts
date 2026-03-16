/**
 * DocumentManager — pure document lifecycle manager with reactive diagnostics.
 *
 * Diagnostics results are Svelte $state — AcvusEngineField can use $derived
 * to automatically react to changes. No imperative refresh needed.
 */

import type { LanguageSession, ContextKeyInfo, CompletionItem, EngineError, DocScope } from './engine.js';
import type { TypeDesc } from './type-parser.js';
import { isUnknownType } from './type-parser.js';

export type ScriptMode = 'script' | 'template';

export type ScriptEntry = {
	source: string;
	mode: ScriptMode;
	scope: DocScope;
};

export type DiagResult = {
	ok: boolean;
	errors: EngineError[];
};

export type CollectedKeys = {
	keys: ContextKeyInfo[];
	hasSkippedScripts: boolean;
};

export type KeyFilter = {
	exclude: Set<string>;
};

interface DocState {
	docId: number;
	source: string;
	mode: ScriptMode;
}

const EMPTY_DIAG: DiagResult = { ok: true, errors: [] };

export class DocumentManager {
	private session: LanguageSession;
	private docs = new Map<string, DocState>();

	/** Reactive diagnostics cache — Svelte tracks property reads/writes on plain objects. */
	_diag: Record<string, DiagResult> = $state({});

	constructor(session: LanguageSession) {
		this.session = session;
	}

	get lspSession(): LanguageSession {
		return this.session;
	}

	has(key: string): boolean {
		return this.docs.has(key);
	}

	docId(key: string): number | undefined {
		return this.docs.get(key)?.docId;
	}

	// -----------------------------------------------------------------------
	// Diff-based sync
	// -----------------------------------------------------------------------

	sync(scripts: Map<string, ScriptEntry>): boolean {
		let changed = false;

		for (const [key, state] of this.docs) {
			if (!scripts.has(key)) {
				this.session.close(state.docId);
				this.docs.delete(key);
				delete this._diag[key];
				changed = true;
			}
		}

		for (const [key, entry] of scripts) {
			const existing = this.docs.get(key);
			if (!existing) {
				const docId = this.session.open(entry.source, entry.mode, entry.scope);
				this.docs.set(key, { docId, source: entry.source, mode: entry.mode });
				this.recomputeDiag(key);
				changed = true;
			} else {
				if (existing.source !== entry.source) {
					this.session.updateSource(existing.docId, entry.source);
					existing.source = entry.source;
					this.recomputeDiag(key);
					changed = true;
				}
				this.session.updateScope(existing.docId, entry.scope);
			}
		}

		return changed;
	}

	// -----------------------------------------------------------------------
	// Per-document mutations
	// -----------------------------------------------------------------------

	updateSource(key: string, source: string): void {
		const state = this.docs.get(key);
		if (!state) return;
		if (state.source === source) return;
		this.session.updateSource(state.docId, source);
		state.source = source;
		this.recomputeDiag(key);
	}

	updateScope(key: string, scope: DocScope): void {
		const state = this.docs.get(key);
		if (!state) return;
		this.session.updateScope(state.docId, scope);
		this.recomputeDiag(key);
	}

	// -----------------------------------------------------------------------
	// Field errors injection (from rebuildNodes)
	// -----------------------------------------------------------------------

	/**
	 * Bind a document to a node field. Diagnostics for bound documents
	 * come from inference (shared subst) via rebuild_nodes, not standalone check.
	 */
	bindDocToNode(key: string, nodeName: string, field: string, fieldIndex?: number): void {
		const state = this.docs.get(key);
		if (!state) return;
		this.session.bindDocToNode(state.docId, nodeName, field, fieldIndex);
	}

	/**
	 * Notify that rebuild_nodes completed — recompute diagnostics for all bound documents.
	 * Inference results are now stored in LspSession, diagnostics(docId) returns them.
	 */
	onRebuildComplete(): void {
		this.recomputeAllDiags();
	}

	// -----------------------------------------------------------------------
	// Reactive queries
	// -----------------------------------------------------------------------

	/**
	 * Get diagnostics for a document. **Reactive** — returns from $state Map.
	 * Svelte $derived will automatically re-evaluate when this changes.
	 */
	diagnostics(key: string): DiagResult {
		return this._diag[key] ?? EMPTY_DIAG;
	}

	contextKeys(key: string, known?: Record<string, string>): ContextKeyInfo[] {
		const state = this.docs.get(key);
		if (!state) return [];
		return this.session.contextKeys(state.docId, known);
	}

	completions(key: string, cursor: number): CompletionItem[] {
		const state = this.docs.get(key);
		if (!state) return [];
		return this.session.completions(state.docId, cursor);
	}

	// -----------------------------------------------------------------------
	// Aggregated queries
	// -----------------------------------------------------------------------

	collectAllKeys(filter: KeyFilter, known?: Record<string, string>): CollectedKeys {
		const seen = new Map<string, { type: TypeDesc; status: 'eager' | 'lazy' | 'pruned' }>();
		let hasSkippedScripts = false;

		for (const [, state] of this.docs) {
			const keys = this.session.contextKeys(state.docId, known);

			if (keys.length === 0 && state.source.trim().length > 0) {
				const diag = this.session.diagnostics(state.docId);
				if (!diag.ok && diag.errors.some((e) => e.category === 'parse')) {
					hasSkippedScripts = true;
				}
			}
			for (const key of keys) {
				if (filter.exclude.has(key.name)) continue;

				const existing = seen.get(key.name);
				if (!existing || isUnknownType(existing.type)) {
					seen.set(key.name, { type: key.type, status: key.status });
				} else if (existing.status === 'pruned' && key.status !== 'pruned') {
					seen.set(key.name, { type: existing.type, status: key.status });
				} else if (existing.status === 'lazy' && key.status === 'eager') {
					seen.set(key.name, { type: existing.type, status: 'eager' });
				}
			}
		}

		const keys = Array.from(seen.entries())
			.map(([name, { type, status }]) => ({ name, type, status }))
			.sort((a, b) => a.name.localeCompare(b.name));

		return { keys, hasSkippedScripts };
	}

	// -----------------------------------------------------------------------
	// Lifecycle
	// -----------------------------------------------------------------------

	dispose(): void {
		for (const state of this.docs.values()) {
			this.session.close(state.docId);
		}
		this.docs.clear();
		this._diag = {};
	}

	// -----------------------------------------------------------------------
	// Internal
	// -----------------------------------------------------------------------

	private recomputeDiag(key: string): void {
		const state = this.docs.get(key);
		if (!state) {
			delete this._diag[key];
			return;
		}
		// LspSession.diagnostics handles bound vs unbound:
		// - bound document → inference result (shared subst, accurate)
		// - unbound document → standalone check_script
		const diag = this.session.diagnostics(state.docId);
		this._diag[key] = diag;
	}

	private recomputeAllDiags(): void {
		for (const key of this.docs.keys()) {
			this.recomputeDiag(key);
		}
	}
}
