<script lang="ts">
	import type { CompletionItem } from '$lib/engine.js';
	import type { DocumentManager } from '$lib/document-manager.svelte.js';
	import { highlightTemplate, highlightScript } from '$lib/highlight.js';

	let {
		value,
		oninput,
		mode = 'script',
		placeholder = '',
		rows = 1,
		unlimited = false,
		docManager,
		docKey,
	}: {
		value: string;
		oninput: (value: string) => void;
		mode?: 'script' | 'template';
		placeholder?: string;
		rows?: number;
		unlimited?: boolean;
		docManager: DocumentManager;
		docKey: string;
	} = $props();

	let hlHtml = $state('');
	let textareaEl = $state<HTMLTextAreaElement | null>(null);
	let hlEl = $state<HTMLDivElement | null>(null);

	let completionItems = $state<CompletionItem[]>([]);
	let showDropdown = $state(false);
	let selectedIndex = $state(0);
	let dropdownTop = $state(0);

	// Diagnostics — fully reactive via $derived.
	// DocumentManager._diag is $state(Map) → .get() is tracked → auto re-evaluate.
	let diagResult = $derived(docManager.diagnostics(docKey));
	let hasError = $derived(!diagResult.ok);
	let displayErrors = $derived(diagResult.ok ? [] : diagResult.errors.map(e => e.message));

	function updateHighlight(source: string) {
		if (mode === 'template') {
			hlHtml = highlightTemplate(source);
		} else {
			hlHtml = highlightScript(source);
		}
	}

	$effect(() => {
		updateHighlight(value);
	});

	function triggerCompletion() {
		if (!textareaEl) return;
		const cursor = textareaEl.selectionStart;
		const items = docManager.completions(docKey, cursor);
		completionItems = items;
		showDropdown = completionItems.length > 0;
		selectedIndex = 0;

		// Position dropdown near cursor
		if (showDropdown) {
			const text = textareaEl.value.slice(0, cursor);
			const lines = text.split('\n');
			const lineHeight = parseFloat(getComputedStyle(textareaEl).lineHeight) || 20;
			dropdownTop = (lines.length * lineHeight) - textareaEl.scrollTop + 4;
		}
	}

	function handleInput(e: Event & { currentTarget: HTMLTextAreaElement }) {
		const v = e.currentTarget.value;
		oninput(v);
		docManager.updateSource(docKey, v);
		// Check for completion triggers
		const lastChar = v[e.currentTarget.selectionStart - 1];
		if (lastChar === '@' || lastChar === '|' || lastChar === '.') {
			triggerCompletion();
		} else if (showDropdown) {
			triggerCompletion(); // Update filter while typing
		}
	}

	function handleKeydown(e: KeyboardEvent) {
		if (!showDropdown) {
			if (e.ctrlKey && e.key === ' ') {
				e.preventDefault();
				triggerCompletion();
			}
			return;
		}
		switch (e.key) {
			case 'ArrowDown':
				e.preventDefault();
				selectedIndex = (selectedIndex + 1) % completionItems.length;
				break;
			case 'ArrowUp':
				e.preventDefault();
				selectedIndex = (selectedIndex - 1 + completionItems.length) % completionItems.length;
				break;
			case 'Enter':
			case 'Tab':
				e.preventDefault();
				applyCompletion(completionItems[selectedIndex]);
				break;
			case 'Escape':
				showDropdown = false;
				break;
		}
	}

	function applyCompletion(item: CompletionItem) {
		if (!textareaEl) return;
		const cursor = textareaEl.selectionStart;
		const before = textareaEl.value.slice(0, cursor);
		const after = textareaEl.value.slice(cursor);

		// Find how much of the prefix to replace.
		// Context completions: replace everything after the last '@'
		// Pipe completions: replace everything after the last '|'
		// Keyword completions: replace the current word
		let replaceStart = cursor;
		if (item.kind === 'context') {
			const atPos = before.lastIndexOf('@');
			if (atPos >= 0) replaceStart = atPos + 1; // keep the '@', replace after it
		} else if (item.kind === 'keyword') {
			// Find start of current word (unicode-safe for future i18n identifiers)
			let i = cursor - 1;
			while (i >= 0 && /[\p{L}\p{N}_]/u.test(before[i])) i--;
			replaceStart = i + 1;
		}
		// For 'builtin' (pipe), insertText already has leading space, cursor is right after '|'

		const prefix = textareaEl.value.slice(0, replaceStart);
		const newValue = prefix + item.insertText + after;
		oninput(newValue);
		docManager.updateSource(docKey, newValue);
		showDropdown = false;
		requestAnimationFrame(() => {
			if (textareaEl) {
				const newPos = replaceStart + item.insertText.length;
				textareaEl.selectionStart = newPos;
				textareaEl.selectionEnd = newPos;
				textareaEl.focus();
			}
		});
	}

	function handleBlur() {
		// Delay to allow click on dropdown
		setTimeout(() => { showDropdown = false; }, 150);
	}

	function handleScroll() {
		if (textareaEl && hlEl) {
			hlEl.scrollTop = textareaEl.scrollTop;
			hlEl.scrollLeft = textareaEl.scrollLeft;
		}
	}

	function autogrow(el: HTMLTextAreaElement, skip: boolean) {
		if (skip) return;
		function resize() {
			el.style.height = 'auto';
			el.style.height = el.scrollHeight + 'px';
		}
		resize();
		el.addEventListener('input', resize);
		return { destroy() { el.removeEventListener('input', resize); } };
	}
</script>

<div class="sf-wrap" class:sf-unlimited={unlimited}>
	<div class="sf-editor" class:sf-unlimited={unlimited} style:--sf-rows={rows}>
		<!-- svelte-ignore a11y_no_static_element_interactions -->
		<div class="sf-hl" bind:this={hlEl} aria-hidden="true">{#if !value && placeholder}<span class="sf-placeholder">{placeholder}</span>{:else}{@html hlHtml}{/if}</div>
		<textarea
			class="sf-textarea"
			class:sf-error={hasError}
			{rows}
			{value}
			oninput={handleInput}
			onblur={handleBlur}
			onkeydown={handleKeydown}
			onscroll={handleScroll}
			spellcheck="false"
			use:autogrow={unlimited}
			bind:this={textareaEl}
		></textarea>
		{#if showDropdown && completionItems.length > 0}
			<div class="sf-completions" style="top: {dropdownTop}px">
				{#each completionItems as item, i}
					<button
						class="sf-completion-item"
						class:sf-selected={i === selectedIndex}
						onmousedown={(e) => { e.preventDefault(); applyCompletion(item); }}
					>
						<span class="sf-completion-label">{item.label}</span>
						{#if item.detail}
							<span class="sf-completion-detail">{item.detail}</span>
						{/if}
					</button>
				{/each}
			</div>
		{/if}
	</div>
	{#each displayErrors as err}
		<p class="sf-error-msg">{err}</p>
	{/each}
</div>

<style>
	.sf-wrap {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
	}
	.sf-wrap.sf-unlimited {
		flex: 1;
		min-height: 0;
	}
	.sf-editor {
		position: relative;
		min-height: calc(var(--sf-rows, 1) * 0.75rem * 1.625 + 0.75rem);
	}
	.sf-editor.sf-unlimited {
		flex: 1;
		min-height: 0;
	}
	.sf-hl,
	.sf-textarea {
		box-sizing: border-box;
		font-family: var(--font-mono, ui-monospace, monospace);
		font-size: 0.75rem;
		line-height: 1.625;
		letter-spacing: normal;
		white-space: pre-wrap;
		word-break: break-all;
		tab-size: 2;
		-moz-tab-size: 2;
		padding: 0.375rem 0.5rem;
		margin: 0;
		border: 1px solid var(--color-border);
		border-radius: 0.375rem;
	}
	.sf-hl {
		position: absolute;
		inset: 0;
		overflow: hidden;
		pointer-events: none;
		z-index: 0;
		color: var(--color-foreground);
		background: transparent;
		scrollbar-width: none;
	}
	.sf-unlimited .sf-hl {
		overflow: auto;
	}
	.sf-hl::-webkit-scrollbar { display: none; }
	.sf-textarea {
		position: relative;
		z-index: 1;
		width: 100%;
		resize: none;
		overflow-y: auto;
		max-height: 24rem;
		background: transparent;
		color: transparent;
		caret-color: var(--color-foreground);
		outline: none;
		transition: border-color 0.15s;
		/* Hide scrollbar so content width matches overlay exactly */
		scrollbar-width: none;
		/* Remove mobile browser default textarea styling */
		-webkit-appearance: none;
		appearance: none;
		-webkit-text-size-adjust: 100%;
	}
	.sf-textarea::-webkit-scrollbar { display: none; }
	.sf-unlimited .sf-textarea {
		position: absolute;
		inset: 0;
		height: 100%;
		max-height: none;
	}
	.sf-textarea::selection {
		background: color-mix(in oklch, var(--color-primary) 30%, transparent);
		color: transparent;
	}
	.sf-textarea:focus {
		box-shadow: 0 0 0 1px var(--color-ring);
	}
	.sf-textarea.sf-error {
		border-color: var(--color-destructive);
	}
	.sf-textarea.sf-error:focus {
		box-shadow: 0 0 0 1px var(--color-destructive);
	}
	.sf-placeholder {
		color: var(--color-muted-foreground);
		pointer-events: none;
		user-select: none;
	}
	.sf-error-msg {
		font-size: 0.6875rem;
		color: var(--color-destructive);
		margin: 0;
	}
	.sf-completions {
		position: absolute;
		z-index: 10;
		left: 0;
		right: 0;
		max-height: 12rem;
		overflow-y: auto;
		background: var(--color-background);
		border: 1px solid var(--color-border);
		border-radius: 0.375rem;
		box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
		margin-top: 2px;
	}
	.sf-completion-item {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		width: 100%;
		padding: 0.25rem 0.5rem;
		font-size: 0.75rem;
		font-family: var(--font-mono, ui-monospace, monospace);
		text-align: left;
		border: none;
		background: transparent;
		cursor: pointer;
		color: var(--color-foreground);
	}
	.sf-completion-item:hover,
	.sf-completion-item.sf-selected {
		background: var(--color-accent);
	}
	.sf-completion-label {
		font-weight: 500;
	}
	.sf-completion-detail {
		color: var(--color-muted-foreground);
		font-size: 0.6875rem;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	/* Syntax highlight tokens */
	.sf-hl :global(.hl-delim) { color: hsl(207 90% 60%); font-weight: 600; }
	.sf-hl :global(.hl-kw) { color: hsl(300 50% 65%); }
	.sf-hl :global(.hl-ctx) { color: hsl(158 60% 55%); }
	.sf-hl :global(.hl-str) { color: hsl(20 70% 60%); }
	.sf-hl :global(.hl-num) { color: hsl(96 40% 65%); }
	.sf-hl :global(.hl-fn) { color: hsl(50 70% 65%); }
	.sf-hl :global(.hl-pipe) { color: hsl(50 70% 65%); font-weight: 600; }
	.sf-hl :global(.hl-op) { color: var(--color-foreground); }
	.sf-hl :global(.hl-raw) { color: var(--color-muted-foreground); }

	/* Markdown tokens */
	.sf-hl :global(.hl-md-h) { color: hsl(174 60% 55%); font-weight: 600; }
	.sf-hl :global(.hl-md-b) { font-weight: 600; }
	.sf-hl :global(.hl-md-i) { font-style: italic; }
	.sf-hl :global(.hl-md-code) { color: hsl(0 60% 60%); }
	.sf-hl :global(.hl-md-link) { color: hsl(207 90% 60%); }
	.sf-hl :global(.hl-md-li) { color: hsl(174 60% 55%); }
	.sf-hl :global(.hl-md-bq) { color: var(--color-muted-foreground); font-style: italic; }
	.sf-hl :global(.hl-md-hr) { color: var(--color-muted-foreground); }
</style>
