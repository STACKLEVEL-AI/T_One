import type { SupportMessage, SupportReply } from '../types';

const API_URL = import.meta.env.VITE_SUPPORT_API_URL ?? 'http://localhost:8001/api';

// ---------- Backend contracts ----------
type BackendMessage = {
  id: string;
  userId: string;
  userName: string;
  message: string;
  category: string;
  subcategory?: string[] | string;
  suggestedResponse?: string[] | string | null;
  priority?: string;
  createdAt: string;
};

const toStringArray = (v?: string[] | string | null): string[] =>
  Array.isArray(v) ? v : v ? [v] : [];

const normalizePriority = (p?: string): 'высокий' | 'средний' =>
  p?.trim().toLowerCase() === 'средний' ? 'средний' : 'высокий';

const adaptToSupportMessage = (b: BackendMessage): SupportMessage => {
  const replies = loadLocalReplies(b.id);
  const suggestions = toStringArray(b.suggestedResponse);
  const subcats = toStringArray(b.subcategory);

  return {
    id: b.id,
    userId: b.userId,
    userName: b.userName,
    userAvatar: initials(b.userName),
    message: b.message,
    timestamp: new Date(b.createdAt).toISOString(),
    status: replies.length ? 'answered' : 'pending',
    category: b.category,
    subcategory: subcats[0] ?? '',
    subcategories: subcats.length ? subcats : undefined,
    priority: normalizePriority(b.priority),
    suggestedResponse: suggestions[0],
    suggestedResponses: suggestions.length ? suggestions : undefined,
    replies,
  };
};

// ---------- LocalStorage helpers ----------
const repliesKey = (messageId: string) => `support_replies_${messageId}`;
const loadLocalReplies = (messageId: string): SupportReply[] => {
  try {
    const raw = localStorage.getItem(repliesKey(messageId));
    return raw ? (JSON.parse(raw) as SupportReply[]) : [];
  } catch {
    return [];
  }
};
const pushLocalReply = (messageId: string, reply: SupportReply) => {
  const arr = loadLocalReplies(messageId);
  arr.push(reply);
  localStorage.setItem(repliesKey(messageId), JSON.stringify(arr));
};

// ---------- Utils ----------
const initials = (fullName: string) => {
  const parts = fullName.trim().split(/\s+/);
  const a = (parts[0]?.[0] ?? '').toUpperCase();
  const b = (parts[1]?.[0] ?? '').toUpperCase();
  return a + b || '??';
};

// ---------- API ----------
export const fetchSupportMessages = async (): Promise<SupportMessage[]> => {
  const res = await fetch(`${API_URL}/messages`, { credentials: 'omit' });
  if (!res.ok) throw new Error(`Failed to fetch messages: ${res.status}`);
  const data: BackendMessage[] = await res.json();
  return data.map(adaptToSupportMessage);
};

export const sendUserMessage = async (message: string): Promise<SupportMessage> => {
  const res = await fetch(`${API_URL}/messages`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'omit',
    body: JSON.stringify({ message }),
  });
  if (!res.ok) throw new Error(`Failed to create message: ${res.status}`);
  const created: BackendMessage = await res.json();
  return adaptToSupportMessage(created);
};

export const sendSupportReply = async ({
  messageId,
  reply,
}: {
  messageId: string;
  reply: string;
}): Promise<SupportReply> => {
  const newReply: SupportReply = {
    id: `r_${Date.now()}`,
    from: 'support',
    message: reply,
    timestamp: new Date().toISOString(),
  };
  pushLocalReply(messageId, newReply);
  await new Promise((r) => setTimeout(r, 200));
  return newReply;
};

export type DatasetRecord = {
  question: string;
  primary: string | null;
  sideRecommendations: string[];
  answer: string;
  matched: boolean;
};

export const sendDatasetRecord = async (payload: DatasetRecord): Promise<void> => {
  const res = await fetch(`${API_URL}/dataset`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'omit',
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`Failed to POST /dataset: ${res.status} ${text}`);
  }
};
