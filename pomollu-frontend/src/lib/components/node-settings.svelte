<script lang="ts">
	import type { Node, MessageDef, Strategy } from '$lib/types.js';
	import { blockLabel } from '$lib/types.js';
	import type { BlockOwner } from '$lib/stores.svelte.js';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';
	import { Button } from '$lib/components/ui/button';
	import * as Select from '$lib/components/ui/select';
	import { getOwnerChildren, updateOwnerNodeItem, uiState, providerStore } from '$lib/stores.svelte.js';
	import { findNodeItem, collectBlocks } from '$lib/block-tree.js';
	import { Plus, Trash2 } from 'lucide-svelte';
	import AcvusEngineField from './acvus-engine-field.svelte';

	let {
		nodeId,
		owner,
		contextTypes = {},
		nodeLocals = {},
		nodeErrors = {},
	}: {
		nodeId: string;
		owner: BlockOwner;
		contextTypes?: Record<string, string>;
		nodeLocals?: Record<string, { raw: string; self: string }>;
		nodeErrors?: Record<string, Record<string, string>>;
	} = $props();

	let node = $derived.by(() => {
		const children = getOwnerChildren(owner);
		return children ? findNodeItem(children, nodeId) : undefined;
	});

	// Merge external context types with this node's local types (@raw, @self)
	let locals = $derived(node ? nodeLocals[node.name] : undefined);
	let mergedContextTypes = $derived(
		locals ? { ...contextTypes, raw: locals.raw, self: locals.self } : contextTypes
	);

	// Per-field errors from orchestration typecheck
	let fieldErrors = $derived(node ? (nodeErrors[node.name] ?? {}) : {});
	let providers = $derived(providerStore.providers);
	let hasOrphanProvider = $derived(
		node?.kind === 'llm' && node.providerId !== '' && !providerStore.get(node.providerId)
	);

	function updateNode(updater: (n: Node) => Node) {
		updateOwnerNodeItem(owner, nodeId, updater);
	}

	function handleRemove() {
		uiState.removeOwnerTreeNode(owner, nodeId);
	}

	// --- Strategy ---

	const strategy = $derived(node?.strategy ?? { mode: 'once-per-turn' as const });

	function setStrategyMode(mode: string) {
		const strategies: Record<string, Strategy> = {
			'always': { mode: 'always' },
			'once-per-turn': { mode: 'once-per-turn' },
			'if-modified': { mode: 'if-modified', key: '' },
			'history': { mode: 'history', historyBind: '@raw' },
		};
		updateNode((n) => ({ ...n, strategy: strategies[mode] }));
	}

	// --- Messages ---

	function addBlockMessage() {
		updateNode((n) => ({
			...n,
			messages: [...(n.messages ?? []), { kind: 'block', role: 'system', source: { type: 'inline', template: '' } }]
		}));
	}

	function addIteratorMessage() {
		updateNode((n) => ({
			...n,
			messages: [...(n.messages ?? []), { kind: 'iterator', iterator: '' }]
		}));
	}

	function updateMessage(index: number, patch: Record<string, unknown>) {
		updateNode((n) => {
			const msgs = [...(n.messages ?? [])];
			msgs[index] = { ...msgs[index], ...patch } as MessageDef;
			return { ...n, messages: msgs };
		});
	}

	function setMessageSource(index: number, source: import('$lib/types.js').MessageSource) {
		updateNode((n) => {
			const msgs = [...(n.messages ?? [])];
			const m = msgs[index];
			if (m.kind !== 'block') return n;
			msgs[index] = { ...m, source };
			return { ...n, messages: msgs };
		});
	}

	function removeMessage(index: number) {
		updateNode((n) => ({
			...n,
			messages: (n.messages ?? []).filter((_, i) => i !== index)
		}));
	}

	function moveMessage(index: number, direction: -1 | 1) {
		updateNode((n) => {
			const msgs = [...(n.messages ?? [])];
			const target = index + direction;
			if (target < 0 || target >= msgs.length) return n;
			[msgs[index], msgs[target]] = [msgs[target], msgs[index]];
			return { ...n, messages: msgs };
		});
	}

	const messages = $derived(node?.messages ?? []);

	let collapsed: Record<string, boolean> = $state({});

	// All blocks available from the owner's tree (for block reference picker)
	const availableBlocks = $derived.by(() => {
		const ownerChildren = getOwnerChildren(owner);
		if (!ownerChildren) return [];
		// Collect from owner children + node's own children
		const blocks = collectBlocks(ownerChildren);
		return blocks;
	});

	function toggleSection(name: string) {
		collapsed[name] = !collapsed[name];
	}

	const roleColors: Record<string, string> = {
		system: 'msg-system',
		user: 'msg-user',
		assistant: 'msg-assistant',
	};

	function msgRole(msg: MessageDef): string | undefined {
		return msg.kind === 'block' ? msg.role : msg.role;
	}

	const strategyLabels: Record<string, string> = {
		'always': 'Always',
		'once-per-turn': 'Once Per Turn',
		'if-modified': 'If Modified',
		'history': 'History',
	};

	const strategyDescriptions: Record<string, string> = {
		'always': 'Execute every invocation. @self is volatile (per-turn).',
		'once-per-turn': 'Execute once per turn. @self is persistent.',
		'if-modified': 'Execute only when key changes. @self is persistent.',
		'history': 'Execute once per turn + append to history.',
	};

	let locked = $derived(uiState.isOwnerBusy(owner));
</script>

<div class="flex h-full flex-col" class:pointer-events-none={locked} class:opacity-60={locked}>
	{#if locked}
		<div class="shrink-0 border-b bg-amber-500/10 px-4 py-1.5 text-xs text-amber-700 dark:text-amber-400">Turn in progress — editing locked</div>
	{/if}
	<div class="flex items-center justify-between shrink-0 border-b px-4 py-2">
		<span class="text-sm font-medium">Node Settings</span>
		<Button variant="ghost" size="icon-sm" class="text-muted-foreground hover:text-destructive" onclick={handleRemove} title="Delete node">
			&times;
		</Button>
	</div>

	{#if node}
		<div class="flex-1 overflow-y-auto">
			<div class="sections">
				<!-- ========== BASIC ========== -->
				<section class="section">
					<button class="section-header" onclick={() => toggleSection('basic')}>
						<svg class="section-chevron" class:collapsed={collapsed['basic']} viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 9l6 6 6-6"/></svg>
						<span>Basic</span>
					</button>
					{#if !collapsed['basic']}<div class="section-body">
						<div class="field">
							<Label>Name</Label>
							<Input
								value={node.name}
								oninput={(e) => updateNode((n) => ({ ...n, name: e.currentTarget.value }))}
							/>
						</div>
						<div class="field">
							<Label>Kind</Label>
							<Select.Root type="single" value={node.kind} onValueChange={(v) => updateNode((n) => ({ ...n, kind: v as 'llm' | 'plain' }))}>
								<Select.Trigger class="w-full">{node.kind === 'llm' ? 'LLM' : 'Plain'}</Select.Trigger>
								<Select.Content>
									<Select.Item value="llm">LLM</Select.Item>
									<Select.Item value="plain">Plain</Select.Item>
								</Select.Content>
							</Select.Root>
						</div>
						{#if node.kind === 'llm'}
							<div class="field">
								<Label>Provider</Label>
								<Select.Root type="single" value={node.providerId} onValueChange={(v) => updateNode((n) => ({ ...n, providerId: v }))}>
									<Select.Trigger class="w-full {hasOrphanProvider ? 'border-destructive' : ''}">
										{providerStore.get(node.providerId)?.name ?? 'Select provider...'}
									</Select.Trigger>
									<Select.Content>
										{#each providers as p (p.id)}
											<Select.Item value={p.id}>{p.name}</Select.Item>
										{/each}
									</Select.Content>
								</Select.Root>
								{#if hasOrphanProvider}
									<p class="text-xs text-destructive">Provider has been deleted.</p>
								{/if}
							</div>
							<div class="field">
								<Label>Model</Label>
								<Input
									value={node.model}
									oninput={(e) => updateNode((n) => ({ ...n, model: e.currentTarget.value }))}
									placeholder="gemini-2.5-flash, gpt-4o, ..."
								/>
							</div>
							<div class="field">
								<Label>Temperature</Label>
								<Input
									type="number"
									step="0.1"
									min="0"
									max="2"
									value={String(node.temperature)}
									oninput={(e) => updateNode((n) => ({ ...n, temperature: Number(e.currentTarget.value) || 0 }))}
								/>
							</div>
							<div class="grid grid-cols-2 gap-2">
								<div class="field">
									<Label>Max Input Tokens</Label>
									<Input
										type="number"
										value={String(node.maxTokens.input)}
										oninput={(e) => updateNode((n) => ({ ...n, maxTokens: { ...n.maxTokens, input: Number(e.currentTarget.value) || 0 } }))}
									/>
								</div>
								<div class="field">
									<Label>Max Output Tokens</Label>
									<Input
										type="number"
										value={String(node.maxTokens.output)}
										oninput={(e) => updateNode((n) => ({ ...n, maxTokens: { ...n.maxTokens, output: Number(e.currentTarget.value) || 0 } }))}
									/>
								</div>
							</div>
						{/if}
					</div>{/if}
				</section>

				<!-- ========== STRATEGY ========== -->
				<section class="section">
					<button class="section-header" onclick={() => toggleSection('strategy')}>
						<svg class="section-chevron" class:collapsed={collapsed['strategy']} viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 9l6 6 6-6"/></svg>
						<span>Strategy</span>
					</button>
					{#if !collapsed['strategy']}<div class="section-body">
						<div class="field">
							<Label>Mode</Label>
							<Select.Root type="single" value={strategy.mode} onValueChange={(v) => setStrategyMode(v)}>
								<Select.Trigger class="w-full">{strategyLabels[strategy.mode]}</Select.Trigger>
								<Select.Content>
									<Select.Item value="always">Always</Select.Item>
									<Select.Item value="once-per-turn">Once Per Turn</Select.Item>
									<Select.Item value="if-modified">If Modified</Select.Item>
									<Select.Item value="history">History</Select.Item>
								</Select.Content>
							</Select.Root>
							<p class="hint">{strategyDescriptions[strategy.mode]}</p>
						</div>
						{#if strategy.mode === 'if-modified'}
							<div class="field">
								<Label>Key</Label>
								<AcvusEngineField
									mode="script"
									placeholder="e.g. @input | to_string"
									value={strategy.key}
									oninput={(v) => updateNode((n) => ({ ...n, strategy: { mode: 'if-modified', key: v } }))}
									contextTypes={mergedContextTypes}
									discoverContext
								/>
								<p class="hint">Script expression. Re-executes when this value changes.</p>
							</div>
						{:else if strategy.mode === 'history'}
							<div class="field">
								<Label>History Bind</Label>
								<AcvusEngineField
									mode="script"
									placeholder="e.g. @raw"
									value={strategy.historyBind}
									oninput={(v) => updateNode((n) => ({ ...n, strategy: { mode: 'history', historyBind: v } }))}
									contextTypes={mergedContextTypes}
									discoverContext
								/>
								<p class="hint">Script that produces each history entry. Appended to @turn.history.&#123;name&#125;.</p>
							</div>
						{/if}

						<div class="field">
							<Label>Initial Value</Label>
							<AcvusEngineField
								mode="script"
								placeholder=''
								value={node.selfSpec?.initialValue ?? ''}
								oninput={(v) => updateNode((n) => ({ ...n, selfSpec: { ...n.selfSpec, initialValue: v } }))}
								contextTypes={mergedContextTypes}
								expectedTailType={locals?.self}
								discoverContext
							/>
							{#if fieldErrors['initialValue']}
								<p class="text-[0.625rem] text-destructive">{fieldErrors['initialValue']}</p>
							{/if}
							<p class="hint">Initial @self value. When set, @self is available in the node body (previous stored value or this initial value on first run). Leave empty to disable @self.</p>
						</div>

						<!-- Retry / Assert -->
						<div class="grid grid-cols-2 gap-2">
							<div class="field">
								<Label>Retry</Label>
								<Input
									type="number"
									min="0"
									value={String(node.retry ?? 0)}
									oninput={(e) => updateNode((n) => ({ ...n, retry: Number(e.currentTarget.value) || 0 }))}
								/>
							</div>
							<div class="field">
								<Label>Assert</Label>
								<AcvusEngineField
									mode="script"
									placeholder="e.g. @self | length > 0"
									value={node.assert ?? ''}
									oninput={(v) => updateNode((n) => ({ ...n, assert: v }))}
									contextTypes={mergedContextTypes}
									expectedTailType="Bool"
									discoverContext
								/>
								{#if fieldErrors['assert']}
									<p class="text-[0.625rem] text-destructive">{fieldErrors['assert']}</p>
								{/if}
							</div>
						</div>
					</div>{/if}
				</section>

				<!-- ========== MESSAGES (LLM only) ========== -->
				{#if node.kind === 'llm'}
					<section class="section">
						<div class="section-header">
							<button class="section-toggle" onclick={() => toggleSection('messages')}>
								<svg class="section-chevron" class:collapsed={collapsed['messages']} viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 9l6 6 6-6"/></svg>
								<span>Messages</span>
							</button>
							<div class="flex gap-1">
								<Button variant="outline" size="sm" class="h-6 text-xs" onclick={addBlockMessage}>
									<Plus class="mr-1 h-3 w-3" />
									Block
								</Button>
								<Button variant="outline" size="sm" class="h-6 text-xs" onclick={addIteratorMessage}>
									<Plus class="mr-1 h-3 w-3" />
									Iterator
								</Button>
							</div>
						</div>
						{#if !collapsed['messages']}<div class="section-body">
							{#if fieldErrors['messages']}
								<p class="text-[0.625rem] text-destructive">{fieldErrors['messages']}</p>
							{/if}
							{#if messages.length === 0}
								<div class="rounded-md border border-dashed p-4 text-center text-xs text-muted-foreground">
									No messages. Add a block or iterator message.
								</div>
							{/if}

							<div class="space-y-2">
								{#each messages as msg, i (i)}
									{@const role = msgRole(msg)}
									<div class="msg-card {role ? roleColors[role] ?? '' : ''}">
										<!-- Header -->
										<div class="flex items-center gap-1.5 px-2 py-1.5">
											<div class="flex flex-col gap-px">
												<button
													class="msg-move"
													disabled={i === 0}
													onclick={() => moveMessage(i, -1)}
													title="Move up"
												>
													<svg class="h-3 w-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 15l-6-6-6 6"/></svg>
												</button>
												<button
													class="msg-move"
													disabled={i === messages.length - 1}
													onclick={() => moveMessage(i, 1)}
													title="Move down"
												>
													<svg class="h-3 w-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 9l6 6 6-6"/></svg>
												</button>
											</div>
											<span class="msg-kind-badge {msg.kind === 'iterator' ? 'msg-kind-iter' : ''}">
												{msg.kind === 'block' ? 'block' : 'iter'}
											</span>
											{#if msg.kind === 'block'}
												<Select.Root type="single" value={msg.role} onValueChange={(v) => updateMessage(i, { role: v })}>
													<Select.Trigger size="sm" class="msg-role-trigger {roleColors[msg.role] ?? ''}">{msg.role}</Select.Trigger>
													<Select.Content>
														<Select.Item value="system">system</Select.Item>
														<Select.Item value="user">user</Select.Item>
														<Select.Item value="assistant">assistant</Select.Item>
													</Select.Content>
												</Select.Root>
											{:else}
												{#if msg.role}
													<Select.Root type="single" value={msg.role} onValueChange={(v) => updateMessage(i, { role: v || undefined })}>
														<Select.Trigger size="sm" class="msg-role-trigger {roleColors[msg.role] ?? ''}">{msg.role}</Select.Trigger>
														<Select.Content>
															<Select.Item value="auto" label="auto">auto</Select.Item>
															<Select.Item value="system">system</Select.Item>
															<Select.Item value="user">user</Select.Item>
															<Select.Item value="assistant">assistant</Select.Item>
														</Select.Content>
													</Select.Root>
												{:else}
													<button
														class="pill-btn"
														onclick={() => updateMessage(i, { role: 'user' })}
													>+ role</button>
												{/if}
											{/if}
											<div class="flex-1"></div>
											<button
												class="rounded p-0.5 text-muted-foreground hover:text-destructive"
												onclick={() => removeMessage(i)}
												title="Remove"
											>
												<Trash2 class="h-3.5 w-3.5" />
											</button>
										</div>

										<!-- Body -->
										<div class="px-2 pb-2">
											{#if msg.kind === 'block'}
												{@const src = msg.source ?? { type: 'inline', template: '' }}
												<div class="space-y-1.5">
													<!-- Source toggle -->
													<div class="flex items-center gap-1.5">
														<button
															class="source-toggle {src.type === 'inline' ? 'active' : ''}"
															onclick={() => setMessageSource(i, { type: 'inline', template: '' })}
														>inline</button>
														<button
															class="source-toggle {src.type === 'block' ? 'active' : ''}"
															onclick={() => setMessageSource(i, { type: 'block', blockId: '' })}
														>block</button>
													</div>
													{#if src.type === 'inline'}
														<AcvusEngineField
															mode="template"
															placeholder="Template content..."
															value={src.template}
															oninput={(v) => setMessageSource(i, { type: 'inline', template: v })}
															contextTypes={mergedContextTypes}
															discoverContext
														/>
													{:else}
														<Select.Root type="single" value={src.blockId} onValueChange={(v) => setMessageSource(i, { type: 'block', blockId: v })}>
															<Select.Trigger class="w-full" size="sm">
																{#if src.blockId}
																	{@const blk = availableBlocks.find(b => b.id === src.blockId)}
																	{blk ? blockLabel(blk) : 'Unknown'}
																{:else}
																	Select block...
																{/if}
															</Select.Trigger>
															<Select.Content>
																{#each availableBlocks as b (b.id)}
																	<Select.Item value={b.id}>{blockLabel(b)}</Select.Item>
																{/each}
																{#if availableBlocks.length === 0}
																	<div class="px-2 py-1.5 text-xs text-muted-foreground">No blocks available</div>
																{/if}
															</Select.Content>
														</Select.Root>
													{/if}
												</div>
											{:else}
												<div class="space-y-1.5">
													<AcvusEngineField
														mode="script"
														placeholder="Iterator expression, e.g. @turn.history | map(h -> h.chat)"
														value={msg.iterator}
														oninput={(v) => updateMessage(i, { iterator: v })}
														contextTypes={mergedContextTypes}
														discoverContext
													/>
													<!-- Slice -->
													{#if msg.slice}
														<div class="flex items-center gap-1.5">
															<span class="text-[10px] text-muted-foreground">slice</span>
															<input
																type="number"
																class="msg-slice-input"
																value={msg.slice[0]}
																oninput={(e) => {
																	const v = parseInt(e.currentTarget.value);
																	if (isNaN(v)) return;
																	const s = msg.slice!;
																	updateMessage(i, { slice: s.length > 1 ? [v, s[1]] : [v] });
																}}
																placeholder="start"
															/>
															{#if msg.slice.length > 1}
																<span class="text-[10px] text-muted-foreground">:</span>
																<input
																	type="number"
																	class="msg-slice-input"
																	value={msg.slice[1]}
																	oninput={(e) => {
																		const v = parseInt(e.currentTarget.value);
																		if (isNaN(v)) return;
																		updateMessage(i, { slice: [msg.slice![0], v] });
																	}}
																	placeholder="end"
																/>
															{:else}
																<button
																	class="pill-btn"
																	onclick={() => updateMessage(i, { slice: [msg.slice![0], -1] })}
																>+ end</button>
															{/if}
															<button
																class="rounded p-0.5 text-muted-foreground hover:text-destructive"
																onclick={() => updateMessage(i, { slice: undefined })}
																title="Remove slice"
															>
																<Trash2 class="h-3 w-3" />
															</button>
														</div>
													{:else}
														<button
															class="pill-btn"
															onclick={() => updateMessage(i, { slice: [0] })}
														>+ slice</button>
													{/if}
													<!-- Template -->
													{#if msg.template !== undefined}
														<AcvusEngineField
															mode="template"
															placeholder="Item template (optional)..."
															value={msg.template ?? ''}
															oninput={(v) => updateMessage(i, { template: v })}
															contextTypes={mergedContextTypes}
															discoverContext
														/>
													{:else}
														<button
															class="pill-btn"
															onclick={() => updateMessage(i, { template: '' })}
														>+ template</button>
													{/if}
												</div>
											{/if}
										</div>
									</div>
								{/each}
							</div>
						</div>{/if}
					</section>
				{/if}
			</div>
		</div>
	{:else}
		<div class="flex flex-1 items-center justify-center text-sm text-muted-foreground">
			No node selected.
		</div>
	{/if}
</div>

<style>
	/* Sections */
	.sections {
		display: flex;
		flex-direction: column;
	}
	.section {
		border-bottom: 1px solid var(--color-border);
	}
	.section:last-child {
		border-bottom: none;
	}
	.section-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 0.5rem 1rem;
		font-size: 0.6875rem;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: var(--color-muted-foreground);
		background: var(--color-muted);
		cursor: pointer;
		user-select: none;
	}
	/* When section-header is a button (Basic, Strategy) */
	button.section-header {
		width: 100%;
		border: none;
		text-align: left;
		justify-content: flex-start;
		gap: 0.375rem;
	}
	/* Toggle part inside Messages header */
	.section-toggle {
		display: flex;
		align-items: center;
		gap: 0.375rem;
		border: none;
		background: none;
		padding: 0;
		font: inherit;
		color: inherit;
		text-transform: inherit;
		letter-spacing: inherit;
		cursor: pointer;
	}
	.section-chevron {
		width: 0.75rem;
		height: 0.75rem;
		transition: transform 0.15s;
		flex-shrink: 0;
	}
	.section-chevron.collapsed {
		transform: rotate(-90deg);
	}
	.section-body {
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
		padding: 0.75rem 1rem;
	}

	/* Field */
	.field {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
	}

	/* Hint text */
	.hint {
		font-size: 0.6875rem;
		color: var(--color-muted-foreground);
		line-height: 1.3;
	}

	/* Source toggle */
	.source-toggle {
		padding: 0.125rem 0.5rem;
		border-radius: 0.25rem;
		border: 1px solid var(--color-border);
		background: transparent;
		font-size: 0.625rem;
		font-weight: 500;
		color: var(--color-muted-foreground);
		cursor: pointer;
		transition: all 0.15s;
	}
	.source-toggle:hover {
		border-color: var(--color-foreground);
		color: var(--color-foreground);
	}
	.source-toggle.active {
		background: var(--color-foreground);
		color: var(--color-background);
		border-color: var(--color-foreground);
	}

	/* Pill button (+ role, + slice, + template) */
	.pill-btn {
		height: 1.5rem;
		border-radius: 9999px;
		border: 1px dashed var(--color-border);
		padding: 0 0.5rem;
		font-size: 0.625rem;
		color: var(--color-muted-foreground);
		background: transparent;
		cursor: pointer;
		transition: all 0.15s;
	}
	.pill-btn:hover {
		border-color: var(--color-foreground);
		color: var(--color-foreground);
	}

	/* Message card */
	.msg-card {
		border-radius: 0.5rem;
		border: 1px solid var(--color-border);
		border-left: 3px solid var(--color-border);
		background: var(--color-card);
		transition: border-color 0.15s;
	}
	.msg-card.msg-system { border-left-color: var(--color-msg-system, #f59e0b); }
	.msg-card.msg-user { border-left-color: var(--color-msg-user, #3b82f6); }
	.msg-card.msg-assistant { border-left-color: var(--color-msg-assistant, #10b981); }

	/* Role trigger */
	:global(.msg-role-trigger) {
		border-radius: 9999px !important;
		font-size: 0.6875rem !important;
		font-weight: 500 !important;
		height: 1.5rem !important;
		padding: 0 0.5rem !important;
		gap: 0.25rem !important;
	}
	:global(.msg-role-trigger.msg-system) {
		color: var(--color-msg-system, #f59e0b) !important;
		border-color: color-mix(in srgb, var(--color-msg-system, #f59e0b) 40%, transparent) !important;
		background: color-mix(in srgb, var(--color-msg-system, #f59e0b) 8%, transparent) !important;
	}
	:global(.msg-role-trigger.msg-user) {
		color: var(--color-msg-user, #3b82f6) !important;
		border-color: color-mix(in srgb, var(--color-msg-user, #3b82f6) 40%, transparent) !important;
		background: color-mix(in srgb, var(--color-msg-user, #3b82f6) 8%, transparent) !important;
	}
	:global(.msg-role-trigger.msg-assistant) {
		color: var(--color-msg-assistant, #10b981) !important;
		border-color: color-mix(in srgb, var(--color-msg-assistant, #10b981) 40%, transparent) !important;
		background: color-mix(in srgb, var(--color-msg-assistant, #10b981) 8%, transparent) !important;
	}

	/* Kind badge */
	.msg-kind-badge {
		border-radius: 0.25rem;
		padding: 0.125rem 0.375rem;
		font-size: 0.625rem;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.03em;
		color: var(--color-muted-foreground);
		background: var(--color-muted);
	}
	.msg-kind-iter {
		color: var(--color-msg-iter, #8b5cf6);
		background: color-mix(in srgb, var(--color-msg-iter, #8b5cf6) 12%, transparent);
	}

	/* Move buttons */
	.msg-move {
		color: var(--color-muted-foreground);
		transition: color 0.1s;
	}
	.msg-move:hover { color: var(--color-foreground); }
	.msg-move:disabled { opacity: 0.25; pointer-events: none; }

	/* Slice input */
	.msg-slice-input {
		width: 4.5rem;
		height: 1.5rem;
		border-radius: 0.25rem;
		border: 1px solid var(--color-border);
		background: transparent;
		padding: 0 0.375rem;
		font-family: var(--font-mono, ui-monospace, monospace);
		font-size: 0.6875rem;
		text-align: center;
	}
	.msg-slice-input:focus {
		outline: none;
		box-shadow: 0 0 0 1px var(--color-ring);
	}

</style>
