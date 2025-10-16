/**
 * Нормализация текста: нижний регистр, убрать пунктуацию, схлопнуть пробелы.
 * Unicode-совместимо (русский/латиница/цифры).
 */
export const normalizeText = (s: string): string =>
  (s ?? '')
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();

/**
 * Быстрый Левенштейн без лишних аллокаций.
 * Возвращает расстояние (0 — полное совпадение).
 */
export const levenshteinDistance = (aRaw: string, bRaw: string): number => {
  let a = aRaw;
  let b = bRaw;
  if (a === b) return 0;
  if (a.length === 0) return b.length;
  if (b.length === 0) return a.length;

  // Гарантируем, что a короче для экономии памяти
  if (a.length > b.length) {
    [a, b] = [b, a];
  }
  const m = a.length;
  const n = b.length;
  const prev = new Array(m + 1);
  const curr = new Array(m + 1);

  for (let i = 0; i <= m; i++) prev[i] = i;

  for (let j = 1; j <= n; j++) {
    curr[0] = j;
    const bj = b.charCodeAt(j - 1);
    for (let i = 1; i <= m; i++) {
      const cost = a.charCodeAt(i - 1) === bj ? 0 : 1;
      const del = prev[i] + 1;
      const ins = curr[i - 1] + 1;
      const sub = prev[i - 1] + cost;
      curr[i] = del < ins ? (del < sub ? del : sub) : ins < sub ? ins : sub;
    }
    // swap
    for (let i = 0; i <= m; i++) prev[i] = curr[i];
  }

  return prev[m];
};

/**
 * Коэффициент похожести по Левенштейну от 0 до 1.
 */
export const similarityRatio = (a: string, b: string): number => {
  const maxLen = Math.max(a.length, b.length);
  if (maxLen === 0) return 1;
  const dist = levenshteinDistance(a, b);
  return 1 - dist / maxLen;
};

export type SimilarityOptions = {
  /**
   * Если короткая строка почти полностью содержится в длинной — считаем совпадением.
   * По умолчанию 0.9 (90%)
   */
  containRatio?: number;
  /**
   * Порог похожести по Левенштейну.
   * По умолчанию 0.9 (90%)
   */
  levenshteinRatio?: number;
};

/**
 * Примерно-синтаксическое сравнение двух текстов ответов (после нормализации).
 * Используется для флага matched.
 */
export const approxEqualBySyntax = (
  x: string,
  y: string,
  opts: SimilarityOptions = {},
): boolean => {
  const { containRatio = 0.9, levenshteinRatio = 0.9 } = opts;

  const a = normalizeText(x);
  const b = normalizeText(y);
  if (!a || !b) return false;
  if (a === b) return true;

  // Случай "почти полное включение" (например, добавлены приветствия/прощания)
  const shorter = a.length <= b.length ? a : b;
  const longer = a.length <= b.length ? b : a;
  if (longer.includes(shorter) && shorter.length / longer.length >= containRatio) {
    return true;
  }

  // Порог по Левенштейну
  return similarityRatio(a, b) >= levenshteinRatio;
};
