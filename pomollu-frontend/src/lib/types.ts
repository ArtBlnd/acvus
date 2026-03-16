import type { TypeDesc } from './type-parser.js';

export type ContentPart = {
	id: string;
	content_type: string;
	value: string;
};

export type BlockInfo = {
	type: string;
	tags: Record<string, string>;
	description: string;
	[key: string]: unknown;
};

export type ContextBlock = {
	kind: 'context';
	id: string;
	name: string;
	info: BlockInfo;
	priority: number;
	content: ContentPart[];
	enabled: boolean;
};

export type RawBlockMode = 'script' | 'template';

export type RawBlock = {
	kind: 'raw';
	id: string;
	name: string;
	text: string;
	mode: RawBlockMode;
};

export type NodeToolBinding = {
	nodeId: string;
	description: string;
};

export type NoneBlock = {
	kind: 'none';
	id: string;
	name: string;
};

export type Block = NoneBlock | ContextBlock | RawBlock;

export type BlockKind = Block['kind'];

export function blockKind(block: Block): BlockKind {
	return block.kind;
}

export function isContextBlock(block: Block): block is ContextBlock {
	return block.kind === 'context';
}

export function isRawBlock(block: Block): block is RawBlock {
	return block.kind === 'raw';
}

export function blockLabel(block: Block): string {
	if (isContextBlock(block)) return block.name || block.info.description || block.info.type || 'Untitled';
	if (isRawBlock(block)) return block.name || 'Untitled';
	return block.name || 'Untitled';
}

export type BlockFolder = {
	id: string;
	name: string;
	enabled: boolean;
	children: BlockNode[];
};

export type BlockNode =
	| { kind: 'block'; block: Block }
	| { kind: 'folder'; folder: BlockFolder }
	| { kind: 'node'; node: Node };


// --- Sidebar tree (generic folder system) ---

export type SidebarNode =
	| { kind: 'item'; id: string }
	| { kind: 'folder'; id: string; name: string; children: SidebarNode[] };

// --- Provider ---

export type ApiKind = 'openai' | 'anthropic' | 'google';

export type Provider = {
	id: string;
	name: string;
	api: ApiKind;
	endpoint: string;
	apiKey: string;
};

// --- Node (LLM call unit) ---

export type NodeKind = 'llm' | 'plain' | 'expr';

export type MaxTokens = {
	input: number;
	output: number;
};

export type TokenBudget = {
	priority: number;
	min?: number;
	max?: number;
};

export type MessageSource =
	| { type: 'inline'; template: string }
	| { type: 'block'; blockId: string };

export type MessageDef =
	| { kind: 'block'; role: string; source: MessageSource }
	| { kind: 'iterator'; iterator: string; role?: string; template?: string; slice?: number[]; tokenBudget?: TokenBudget };

export type Execution =
	| { mode: 'always' }
	| { mode: 'once-per-turn' }
	| { mode: 'if-modified'; key: string };

export type Persistency =
	| { kind: 'ephemeral' }
	| { kind: 'snapshot' }
	| { kind: 'sequence'; bind: string }
	| { kind: 'patch'; bind: string };

export type FnParam = {
	name: string;
	type: string;
	description?: string;
};

export type Strategy = {
	execution: Execution;
	persistency: Persistency;
	initialValue: string;
	retry: number;
	assert: string;
};

export type ThinkingConfig =
	| { kind: 'off' }
	| { kind: 'low' }
	| { kind: 'medium' }
	| { kind: 'high' }
	| { kind: 'custom'; value: number };

export type Node = {
	id: string;
	name: string;
	kind: NodeKind;
	providerId: string;
	model: string;
	temperature: number;
	topP: number | null;
	topK: number | null;
	grounding: boolean;
	thinking: ThinkingConfig | null;
	maxTokens: MaxTokens;
	strategy: Strategy;
	exprSource: string;
	messages: MessageDef[];
	tools: NodeToolBinding[];
	isFunction: boolean;
	fnParams: FnParam[];
	children: BlockNode[];
};

// --- Context parameter resolution ---

export type ParamResolution =
	| { kind: 'static'; value: string }
	| { kind: 'dynamic' }
	| { kind: 'unresolved' };

export type ContextParam = {
	name: string;
	inferredType: TypeDesc;
	resolution: ParamResolution;
	userType?: TypeDesc;
	/** Whether this param is currently referenced in scripts. Default true. */
	active?: boolean;
	/** Persisted editor preference: 'structured' (default) or 'raw'. */
	editorMode?: 'structured' | 'raw';
};

export type ParamOverride = {
	resolution: ParamResolution;
	userType?: TypeDesc;
	editorMode?: 'structured' | 'raw';
};

// --- Prompt (project) ---

export type ContextBinding = {
	name: string;
	script: string;
};

export const HISTORY_BINDING_NAME = 'history';
export const HISTORY_ENTRY_TYPE: TypeDesc = {
	kind: 'list',
	elem: {
		kind: 'object',
		fields: [
			{ name: 'content', type: { kind: 'primitive', name: 'string' } },
			{ name: 'content_type', type: { kind: 'primitive', name: 'string' } },
			{ name: 'role', type: { kind: 'primitive', name: 'string' } },
		],
	},
};

/** Type of a tag entry: {key: String, value: String}. */
const TAG_ENTRY_TYPE: TypeDesc = {
	kind: 'object',
	fields: [
		{ name: 'key', type: { kind: 'primitive', name: 'string' } },
		{ name: 'value', type: { kind: 'primitive', name: 'string' } },
	],
};

/** Type of each entry in @context lists. */
const CONTEXT_ENTRY_TYPE: TypeDesc = {
	kind: 'object',
	fields: [
		{ name: 'name', type: { kind: 'primitive', name: 'string' } },
		{ name: 'description', type: { kind: 'primitive', name: 'string' } },
		{ name: 'tags', type: { kind: 'list', elem: TAG_ENTRY_TYPE } },
		{ name: 'content', type: { kind: 'primitive', name: 'string' } },
		{ name: 'content_type', type: { kind: 'primitive', name: 'string' } },
	],
};

/** Type of each entry in @context.custom. */
const CONTEXT_CUSTOM_ENTRY_TYPE: TypeDesc = {
	kind: 'object',
	fields: [
		{ name: 'name', type: { kind: 'primitive', name: 'string' } },
		{ name: 'description', type: { kind: 'primitive', name: 'string' } },
		{ name: 'tags', type: { kind: 'list', elem: TAG_ENTRY_TYPE } },
		{ name: 'content', type: { kind: 'primitive', name: 'string' } },
		{ name: 'content_type', type: { kind: 'primitive', name: 'string' } },
		{ name: 'type', type: { kind: 'primitive', name: 'string' } },
	],
};

/** Fixed type of @context — always available, lists are empty by default. */
export const CONTEXT_TYPE: TypeDesc = {
	kind: 'object',
	fields: [
		{ name: 'system', type: { kind: 'list', elem: CONTEXT_ENTRY_TYPE } },
		{ name: 'character', type: { kind: 'list', elem: CONTEXT_ENTRY_TYPE } },
		{ name: 'world_info', type: { kind: 'list', elem: CONTEXT_ENTRY_TYPE } },
		{ name: 'lorebook', type: { kind: 'list', elem: CONTEXT_ENTRY_TYPE } },
		{ name: 'memory', type: { kind: 'list', elem: CONTEXT_ENTRY_TYPE } },
		{ name: 'custom', type: { kind: 'list', elem: CONTEXT_CUSTOM_ENTRY_TYPE } },
		{ name: 'bot_name', type: { kind: 'primitive', name: 'string' } },
	],
};

export type Prompt = {
	id: string;
	name: string;
	children: BlockNode[];
	contextBindings: ContextBinding[];
	paramOverrides: Record<string, ParamOverride>;
	/** Which dynamic param receives the chat input each turn. */
	inputParam?: string;
};

// --- Profile ---

export type Profile = {
	id: string;
	name: string;
	children: BlockNode[];
	paramOverrides: Record<string, ParamOverride>;
};

// --- Bot ---

export type DisplayEntry = {
	id: string;
	name: string;
	condition: string;
	template: string;
};

export type BotDisplay = {
	iterator: string;
	entries: DisplayEntry[];
};

export type Bot = {
	id: string;
	name: string;
	promptId: string;
	profileId: string;
	children: BlockNode[];
	display: BotDisplay;
	regions: DisplayRegion[];
	layout: GridLayout;
	paramOverrides: Record<string, ParamOverride>;
	embeddedStyle?: string;
};

// --- Display region ---

export type DisplayRegionBase = {
	id: string;
	name: string;
};

export type IterableDisplayRegion = DisplayRegionBase & {
	kind: 'iterable';
	iterator: string;
	entries: DisplayEntry[];
};

export type StaticDisplayRegion = DisplayRegionBase & {
	kind: 'static';
	template: string;
};

export type DisplayRegion = IterableDisplayRegion | StaticDisplayRegion;

// --- Grid layout ---

export const GRID_HISTORY = 'history';

export type GridLayout = {
	areas: string[][]; // [row][col] — GRID_HISTORY | regionId | '' (empty)
	colSizes: number[]; // percentages per column
	rowSizes: number[]; // percentages per row
	aspect: number; // width / height ratio (e.g. 16/9 = 1.78, 1 = square)
};

export function createDefaultLayout(): GridLayout {
	return { areas: [[GRID_HISTORY]], colSizes: [100], rowSizes: [100], aspect: 0 };
}

export function gridStyle(layout: GridLayout): string {
	return `grid-template-columns: ${layout.colSizes.map((s) => s + '%').join(' ')}; grid-template-rows: ${layout.rowSizes.map((s) => s + '%').join(' ')};`;
}

export function cumulativeBoundaries(sizes: number[]): number[] {
	const b = [0];
	for (const s of sizes) b.push(b[b.length - 1] + s);
	return b;
}

export type AreaPlacement = {
	id: string;
	colStart: number;
	colEnd: number;
	rowStart: number;
	rowEnd: number;
};

export function computePlacements(layout: GridLayout): AreaPlacement[] {
	const rows = layout.areas.length;
	const cols = layout.colSizes.length;
	const visited = new Set<number>();
	const result: AreaPlacement[] = [];

	for (let r = 0; r < rows; r++) {
		for (let c = 0; c < cols; c++) {
			const id = layout.areas[r]?.[c] ?? '';
			if (!id || visited.has(r * cols + c)) continue;

			let maxCol = c;
			while (maxCol + 1 < cols && layout.areas[r][maxCol + 1] === id) maxCol++;

			let maxRow = r;
			outer: while (maxRow + 1 < rows) {
				for (let cc = c; cc <= maxCol; cc++) {
					if (layout.areas[maxRow + 1]?.[cc] !== id) break outer;
				}
				maxRow++;
			}

			for (let rr = r; rr <= maxRow; rr++) {
				for (let cc = c; cc <= maxCol; cc++) {
					visited.add(rr * cols + cc);
				}
			}

			result.push({ id, colStart: c + 1, colEnd: maxCol + 2, rowStart: r + 1, rowEnd: maxRow + 2 });
		}
	}
	return result;
}

// --- Display result (from engine) ---

export type RenderedCard = {
	name: string;
	content: string;
};

export type RenderedRegion = {
	id: string;
	name: string;
	cards: RenderedCard[];
};

// --- Session ---

export type Session = {
	id: string;
	name: string;
	botId: string;
};
