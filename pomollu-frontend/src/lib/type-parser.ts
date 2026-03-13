// ---------------------------------------------------------------------------
// TypeDesc — structured type representation
// ---------------------------------------------------------------------------

export type TypeDescOrigin =
	| { kind: 'concrete'; id: number }
	| { kind: 'var'; id: number };

export type TypeDesc =
	| { kind: 'primitive'; name: 'String' | 'Int' | 'Float' | 'Bool' }
	| { kind: 'option'; inner: TypeDesc }
	| { kind: 'object'; fields: { name: string; type: TypeDesc }[] }
	| { kind: 'list'; elem: TypeDesc }
	| { kind: 'deque'; elem: TypeDesc; origin: TypeDescOrigin }
	| { kind: 'enum'; name: string; variants: { tag: string; hasPayload: boolean; payloadType?: TypeDesc }[] }
	| { kind: 'unsupported'; raw: string };

// ---------------------------------------------------------------------------
// StructuredValue — editable value tree
// ---------------------------------------------------------------------------

export type StructuredValue =
	| { kind: 'primitive'; value: string }
	| { kind: 'option-some'; inner: StructuredValue }
	| { kind: 'option-none' }
	| { kind: 'enum-variant'; tag: string; payload?: StructuredValue }
	| { kind: 'object'; fields: Record<string, StructuredValue> }
	| { kind: 'raw'; script: string };

// ---------------------------------------------------------------------------
// Parser helpers (mirrors pomollu-engine/src/lib.rs logic)
// ---------------------------------------------------------------------------

function findMatchingClose(s: string, open: number): number {
	let depth = 0;
	for (let i = open; i < s.length; i++) {
		const c = s[i];
		if (c === '<' || c === '{' || c === '(') depth++;
		else if (c === '>' || c === '}' || c === ')') {
			depth--;
			if (depth === 0) return i;
		}
	}
	return -1;
}

function splitTopLevel(s: string, sep: string): string[] {
	const parts: string[] = [];
	let depth = 0;
	let start = 0;
	for (let i = 0; i < s.length; i++) {
		const c = s[i];
		if (c === '<' || c === '{' || c === '(') depth++;
		else if (c === '>' || c === '}' || c === ')') depth--;
		else if (c === sep && depth === 0) {
			parts.push(s.slice(start, i));
			start = i + 1;
		}
	}
	parts.push(s.slice(start));
	return parts;
}

// ---------------------------------------------------------------------------
// parseTypeDesc
// ---------------------------------------------------------------------------

const PRIMITIVES = new Set(['String', 'Int', 'Float', 'Bool']);

export function parseTypeDesc(
	typeStr: string,
): TypeDesc {
	const s = typeStr.trim();
	if (!s || s === '?') return { kind: 'unsupported', raw: s };

	if (PRIMITIVES.has(s)) {
		return { kind: 'primitive', name: s as 'String' | 'Int' | 'Float' | 'Bool' };
	}

	if (s.startsWith('Option<')) {
		const close = findMatchingClose(s, 6);
		if (close === s.length - 1) {
			const inner = s.slice(7, close);
			return { kind: 'option', inner: parseTypeDesc(inner) };
		}
	}

	if (s.startsWith('List<')) {
		const close = findMatchingClose(s, 4);
		if (close === s.length - 1) {
			const inner = s.slice(5, close);
			return { kind: 'list', elem: parseTypeDesc(inner) };
		}
	}

	if (s.startsWith('{') && s.endsWith('}')) {
		const inner = s.slice(1, -1).trim();
		if (!inner) return { kind: 'object', fields: [] };
		const fields: { name: string; type: TypeDesc }[] = [];
		for (const pair of splitTopLevel(inner, ',')) {
			const trimmed = pair.trim();
			if (!trimmed) continue;
			const colon = trimmed.indexOf(':');
			if (colon === -1) return { kind: 'unsupported', raw: s };
			const name = trimmed.slice(0, colon).trim();
			const tyStr = trimmed.slice(colon + 1).trim();
			fields.push({ name, type: parseTypeDesc(tyStr) });
		}
		return { kind: 'object', fields };
	}

	// Enum<Name { Variant1, Variant2(PayloadType) }>
	if (s.startsWith('Enum<')) {
		const close = findMatchingClose(s, 4);
		if (close === s.length - 1) {
			const inner = s.slice(5, close).trim();
			const braceOpen = inner.indexOf('{');
			if (braceOpen !== -1) {
				const name = inner.slice(0, braceOpen).trim();
				const braceClose = inner.lastIndexOf('}');
				const variantsStr = inner.slice(braceOpen + 1, braceClose).trim();
				const variants: { tag: string; hasPayload: boolean; payloadType?: TypeDesc }[] = [];
				if (variantsStr) {
					for (const part of splitTopLevel(variantsStr, ',')) {
						const trimmed = part.trim();
						if (!trimmed) continue;
						const parenIdx = trimmed.indexOf('(');
						if (parenIdx !== -1) {
							const parenClose = trimmed.lastIndexOf(')');
							const payloadStr = trimmed.slice(parenIdx + 1, parenClose).trim();
							const payloadType = payloadStr ? parseTypeDesc(payloadStr) : undefined;
							variants.push({ tag: trimmed.slice(0, parenIdx).trim(), hasPayload: true, payloadType });
						} else {
							variants.push({ tag: trimmed, hasPayload: false });
						}
					}
				}
				return { kind: 'enum', name, variants };
			}
		}
	}

	return { kind: 'unsupported', raw: s };
}

// ---------------------------------------------------------------------------
// isUnknownType — check if a TypeDesc represents the unknown '?' type
// ---------------------------------------------------------------------------

export function isUnknownType(desc: TypeDesc): boolean {
	return desc.kind === 'unsupported' && desc.raw === '?';
}

// ---------------------------------------------------------------------------
// typeDescToString — display a TypeDesc as a human-readable string
// ---------------------------------------------------------------------------

export function typeDescToString(desc: TypeDesc): string {
	switch (desc.kind) {
		case 'primitive':
			return desc.name;
		case 'option':
			return `Option<${typeDescToString(desc.inner)}>`;
		case 'list':
			return `List<${typeDescToString(desc.elem)}>`;
		case 'deque':
			return `Deque<${typeDescToString(desc.elem)}>`;
		case 'object': {
			const fields = desc.fields.map((f) => `${f.name}: ${typeDescToString(f.type)}`);
			return `{${fields.join(', ')}}`;
		}
		case 'enum': {
			const variants = desc.variants.map((v) => {
				if (v.hasPayload && v.payloadType) {
					return `${v.tag}(${typeDescToString(v.payloadType)})`;
				}
				return v.hasPayload ? `${v.tag}(?)` : v.tag;
			});
			return `Enum<${desc.name} { ${variants.join(', ')} }>`;
		}
		case 'unsupported':
			return desc.raw || '?';
	}
}

// ---------------------------------------------------------------------------
// isStructured — can this type be edited with the structured editor?
// ---------------------------------------------------------------------------

export function isStructured(desc: TypeDesc): boolean {
	return desc.kind === 'primitive'
		|| desc.kind === 'option'
		|| desc.kind === 'object'
		|| desc.kind === 'enum';
}

// ---------------------------------------------------------------------------
// createDefaultValue
// ---------------------------------------------------------------------------

export function createDefaultValue(desc: TypeDesc): StructuredValue {
	switch (desc.kind) {
		case 'primitive':
			switch (desc.name) {
				case 'String': return { kind: 'primitive', value: '' };
				case 'Int': return { kind: 'primitive', value: '0' };
				case 'Float': return { kind: 'primitive', value: '0.0' };
				case 'Bool': return { kind: 'primitive', value: 'false' };
			}
			break; // unreachable but satisfies TS
		case 'option':
			return { kind: 'option-none' };
		case 'object': {
			const fields: Record<string, StructuredValue> = {};
			for (const f of desc.fields) {
				fields[f.name] = createDefaultValue(f.type);
			}
			return { kind: 'object', fields };
		}
		case 'enum':
			if (desc.variants.length > 0) {
				const first = desc.variants[0];
				const payload = first.hasPayload && first.payloadType
					? createDefaultValue(first.payloadType)
					: undefined;
				return { kind: 'enum-variant', tag: first.tag, payload };
			}
			return { kind: 'raw', script: '' };
		case 'list':
		case 'deque':
		case 'unsupported':
			return { kind: 'raw', script: '' };
	}
	return { kind: 'raw', script: '' };
}

// ---------------------------------------------------------------------------
// generateScript — StructuredValue → acvus script expression
// ---------------------------------------------------------------------------

function escapeAcvusString(s: string): string {
	return s.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
}

function unescapeAcvusString(s: string): string {
	let result = '';
	for (let i = 0; i < s.length; i++) {
		if (s[i] === '\\' && i + 1 < s.length) {
			const next = s[i + 1];
			if (next === '\\' || next === '"') {
				result += next;
				i++;
				continue;
			}
		}
		result += s[i];
	}
	return result;
}

export function generateScript(value: StructuredValue, desc: TypeDesc): string {
	switch (value.kind) {
		case 'primitive': {
			if (desc.kind !== 'primitive') return value.value;
			switch (desc.name) {
				case 'String': return `"${escapeAcvusString(value.value)}"`;
				case 'Int':
				case 'Float':
					return value.value || '0';
				case 'Bool':
					return value.value === 'true' ? 'true' : 'false';
			}
			return value.value;
		}
		case 'option-none':
			return 'None';
		case 'option-some': {
			if (desc.kind !== 'option') return 'None';
			const inner = generateScript(value.inner, desc.inner);
			return `Some(${inner})`;
		}
		case 'enum-variant': {
			const enumName = desc.kind === 'enum' ? desc.name : '';
			const qualified = enumName ? `${enumName}::${value.tag}` : value.tag;
			if (value.payload) {
				const variantDef = desc.kind === 'enum'
					? desc.variants.find((v) => v.tag === value.tag)
					: undefined;
				const payloadDesc: TypeDesc = variantDef?.payloadType ?? { kind: 'unsupported', raw: '' };
				const payloadScript = generateScript(value.payload, payloadDesc);
				return `${qualified}(${payloadScript})`;
			}
			return qualified;
		}
		case 'object': {
			if (desc.kind !== 'object') return '{}';
			const parts = desc.fields.map((f) => {
				const fieldVal = value.fields[f.name] ?? createDefaultValue(f.type);
				return `${f.name}: ${generateScript(fieldVal, f.type)}`;
			});
			if (parts.length === 0) return '{}';
			return `{${parts.join(', ')},}`;
		}
		case 'raw':
			return value.script;
	}
}

// ---------------------------------------------------------------------------
// parseScript — acvus script expression → StructuredValue (inverse of generateScript)
// Returns null if the script cannot be parsed for the given type.
// ---------------------------------------------------------------------------

export function parseScript(script: string, desc: TypeDesc): StructuredValue | null {
	const s = script.trim();
	if (!s) return null;

	switch (desc.kind) {
		case 'primitive':
			return parsePrimitive(s, desc.name);
		case 'option':
			return parseOptionScript(s, desc);
		case 'enum':
			return parseEnumScript(s, desc);
		case 'object':
			return parseObjectScript(s, desc);
		default:
			return null;
	}
}

function parsePrimitive(s: string, name: string): StructuredValue | null {
	switch (name) {
		case 'String':
			if (s.startsWith('"') && s.endsWith('"') && s.length >= 2) {
				return { kind: 'primitive', value: unescapeAcvusString(s.slice(1, -1)) };
			}
			return null;
		case 'Int':
			if (/^-?\d+$/.test(s)) return { kind: 'primitive', value: s };
			return null;
		case 'Float':
			if (/^-?\d+(\.\d+)?$/.test(s)) return { kind: 'primitive', value: s };
			return null;
		case 'Bool':
			if (s === 'true' || s === 'false') return { kind: 'primitive', value: s };
			return null;
	}
	return null;
}

function parseOptionScript(s: string, desc: { kind: 'option'; inner: TypeDesc }): StructuredValue | null {
	if (s === 'None') return { kind: 'option-none' };
	if (s.startsWith('Some(') && s.endsWith(')')) {
		const inner = parseScript(s.slice(5, -1), desc.inner);
		if (inner) return { kind: 'option-some', inner };
	}
	return null;
}

function parseEnumScript(
	s: string,
	desc: { kind: 'enum'; name: string; variants: { tag: string; hasPayload: boolean; payloadType?: TypeDesc }[] },
): StructuredValue | null {
	const prefix = desc.name ? `${desc.name}::` : '';
	for (const variant of desc.variants) {
		const qualified = prefix + variant.tag;
		if (s === qualified) {
			return { kind: 'enum-variant', tag: variant.tag };
		}
		if (s.startsWith(qualified + '(') && s.endsWith(')')) {
			const payloadStr = s.slice(qualified.length + 1, -1);
			const payload = variant.payloadType ? parseScript(payloadStr, variant.payloadType) : null;
			return { kind: 'enum-variant', tag: variant.tag, payload: payload ?? undefined };
		}
	}
	return null;
}

function parseObjectScript(
	s: string,
	desc: { kind: 'object'; fields: { name: string; type: TypeDesc }[] },
): StructuredValue | null {
	if (!s.startsWith('{') || !s.endsWith('}')) return null;
	const inner = s.slice(1, -1).trim();
	if (!inner) return { kind: 'object', fields: {} };

	const parts = splitTopLevel(inner, ',').filter((p) => p.trim());
	const fields: Record<string, StructuredValue> = {};
	for (const part of parts) {
		const colonIdx = part.indexOf(':');
		if (colonIdx === -1) return null;
		const fieldName = part.slice(0, colonIdx).trim();
		const fieldValue = part.slice(colonIdx + 1).trim();
		const fieldDef = desc.fields.find((f) => f.name === fieldName);
		if (!fieldDef) continue;
		const parsed = parseScript(fieldValue, fieldDef.type);
		if (!parsed) return null;
		fields[fieldName] = parsed;
	}
	return { kind: 'object', fields };
}
