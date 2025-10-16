import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  Send,
  MessageSquare,
  User,
  Clock,
  Tag,
  CheckCircle2,
  AlertCircle,
  Sparkles,
  ChevronDown,
  Copy,
  Check,
} from 'lucide-react';
import { fetchSupportMessages, sendSupportReply, sendDatasetRecord } from '../api/support';
import type { SupportMessage } from '../types';
import { formatTime } from '../utils/formatTime';
import { approxEqualBySyntax } from '../utils/textSimilarity';

const VTB_GRADIENT_R = 'bg-gradient-to-r from-[#009FDF] to-[#0A2973]';
const VTB_GRADIENT_BR = 'bg-gradient-to-br from-[#009FDF] to-[#0A2973]';
const VTB_LIGHT_GRADIENT_R = 'bg-gradient-to-r from-[#E6F7FF] to-[#EEF3FF]';

const SupportAdminInterface: React.FC = () => {
  const [selectedMessage, setSelectedMessage] = useState<string | null>(null);
  const [replyText, setReplyText] = useState('');
  const [showAllSuggestions, setShowAllSuggestions] = useState(false);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
  const replyRef = useRef<HTMLTextAreaElement | null>(null);
  const queryClient = useQueryClient();

  const { data: messages, isLoading } = useQuery<SupportMessage[]>({
    queryKey: ['supportMessages'],
    queryFn: fetchSupportMessages,
  });

  const selectedMessageData = messages?.find((m) => m.id === selectedMessage);

  const autoResize = useCallback(() => {
    const el = replyRef.current;
    if (!el) return;
    el.style.height = 'auto';
    const max = Math.round(window.innerHeight * 0.5);
    const newHeight = Math.min(el.scrollHeight, max);
    el.style.height = `${newHeight}px`;
    el.style.overflowY = el.scrollHeight > newHeight ? 'auto' : 'hidden';
  }, []);

  useEffect(() => {
    autoResize();
  }, [replyText, selectedMessage, autoResize]);

  useEffect(() => {
    const onResize = () => autoResize();
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, [autoResize]);

  // сбрасываем индикатор «скопировано» при переключении треда
  useEffect(() => {
    setCopiedIndex(null);
  }, [selectedMessage]);

  const priorityBadgeClass = (p: SupportMessage['priority']) =>
    p === 'средний'
      ? 'bg-gradient-to-r from-amber-400 to-amber-600 text-white'
      : 'bg-gradient-to-r from-rose-500 to-red-600 text-white';

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      try {
        // Фолбэк для небезопасных контекстов
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.left = '-9999px';
        document.body.appendChild(ta);
        ta.focus();
        ta.select();
        const ok = document.execCommand('copy');
        document.body.removeChild(ta);
        return ok;
      } catch {
        return false;
      }
    }
  };

  const handleCopySuggestion = async (text: string, idx: number) => {
    const ok = await copyToClipboard(text);
    if (ok) {
      setCopiedIndex(idx);
      window.setTimeout(() => setCopiedIndex(null), 1500);
    }
  };

  const handleSuggestedResponseClick = () => {
    if (selectedMessageData?.suggestedResponse) {
      setReplyText(selectedMessageData.suggestedResponse);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter') {
      if (e.shiftKey) return;
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        handleSendReply();
        return;
      }
      e.preventDefault();
      handleSendReply();
    }
  };

  // ---------- Мутация отправки + логирование датасета ----------
  const replyMutation = useMutation({
    mutationFn: (vars: { messageId: string; reply: string }) => sendSupportReply(vars),
    onSuccess: async (_data, vars) => {
      const msg = messages?.find((m) => m.id === vars.messageId);
      if (msg) {
        const allSuggestions = msg.suggestedResponses?.length
          ? msg.suggestedResponses
          : msg.suggestedResponse
            ? [msg.suggestedResponse]
            : [];

        const primary = allSuggestions[0] ?? null;
        const side = allSuggestions.slice(1);

        const used = allSuggestions.some((s) => approxEqualBySyntax(vars.reply, s));

        try {
          await sendDatasetRecord({
            question: msg.message,
            primary,
            sideRecommendations: side,
            answer: vars.reply,
            matched: used,
          });
        } catch (err) {
          // Не блокируем UX, просто лог
          console.warn('Failed to send /dataset record', err);
        }
      }

      // как и раньше — обновляем список и чистим инпут
      queryClient.invalidateQueries({ queryKey: ['supportMessages'] });
      setReplyText('');
      requestAnimationFrame(() => autoResize());
    },
  });

  const handleSendReply = () => {
    if (selectedMessage && replyText.trim()) {
      replyMutation.mutate({ messageId: selectedMessage, reply: replyText.trim() });
    }
  };

  return (
    <div className="h-screen bg-gray-50 flex">
      {/* Sidebar with messages list */}
      <div className="w-96 bg-white border-r border-gray-200 flex flex-col">
        {/* Header */}
        <div className={`p-4 border-b border-gray-200 ${VTB_GRADIENT_R}`}>
          <h1 className="text-xl font-bold text-white mb-1">Панель поддержки</h1>
          <p className="text-sm text-white/70">Активные обращения</p>
        </div>

        {/* Messages list */}
        <div className="flex-1 overflow-y-auto">
          {isLoading ? (
            <div className="flex items-center justify-center h-full">
              <div className="w-8 h-8 border-2 border-[#0A2973] border-t-transparent rounded-full animate-spin" />
            </div>
          ) : (
            <div className="divide-y divide-gray-100">
              {messages?.map((msg) => (
                <button
                  key={msg.id}
                  onClick={() => {
                    setSelectedMessage(msg.id);
                    setShowAllSuggestions(false);
                  }}
                  className={`w-full p-4 text-left hover:bg-gray-50 transition-colors ${
                    selectedMessage === msg.id ? 'bg-blue-50 border-l-4 border-[#0A2973]' : ''
                  }`}
                >
                  <div className="flex items-start gap-3">
                    <div
                      className={`w-10 h-10 rounded-full ${VTB_GRADIENT_BR} flex items-center justify-center text-white font-medium flex-shrink-0 text-sm`}
                    >
                      {msg.userAvatar}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-semibold text-gray-900 text-sm">{msg.userName}</span>

                        {/* PRIORITY BADGE + pending dot */}
                        <div className="flex items-center gap-2">
                          <span
                            className={`px-2 py-0.5 text-[10px] font-semibold rounded-full shadow-sm capitalize ${priorityBadgeClass(msg.priority)}`}
                            title={`Приоритет: ${msg.priority}`}
                          >
                            {msg.priority}
                          </span>
                          {msg.status === 'pending' && (
                            <span className="w-2 h-2 bg-orange-500 rounded-full animate-pulse" />
                          )}
                        </div>
                      </div>

                      <p className="text-sm text-gray-600 truncate mb-2">{msg.message}</p>
                      <div className="flex items-center gap-2 text-xs text-gray-500">
                        <Clock className="w-3 h-3" />
                        {formatTime(msg.timestamp)}
                      </div>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Main content area */}
      <div className="flex-1 flex flex-col">
        {selectedMessageData ? (
          <>
            {/* Message header */}
            <div className="bg-white border-b border-gray-200 p-4">
              <div className="flex items-center gap-3">
                <div
                  className={`w-12 h-12 rounded-full ${VTB_GRADIENT_BR} flex items-center justify-center text-white font-medium`}
                >
                  {selectedMessageData.userAvatar}
                </div>
                <div className="flex-1">
                  <h2 className="font-semibold text-gray-900">{selectedMessageData.userName}</h2>
                  <p className="text-sm text-gray-500">{selectedMessageData.userId}</p>
                </div>
                <div className="flex items-center gap-2">
                  {selectedMessageData.status === 'pending' ? (
                    <span className="px-3 py-1 bg-orange-100 text-orange-700 text-sm font-medium rounded-full flex items-center gap-1">
                      <AlertCircle className="w-3 h-3" />
                      Ожидает ответа
                    </span>
                  ) : (
                    <span className="px-3 py-1 bg-green-100 text-green-700 text-sm font-medium rounded-full flex items-center gap-1">
                      <CheckCircle2 className="w-3 h-3" />
                      Отвечено
                    </span>
                  )}
                </div>
              </div>

              {/* Category tags */}
              <div className="flex items-center gap-2 mt-3">
                <Tag className="w-4 h-4 text-gray-400" />
                <span className="px-2 py-1 bg-blue-100 text-blue-700 text-xs font-medium rounded">
                  Категория: {selectedMessageData.category}
                </span>
                <span className="px-2 py-1 bg-purple-100 text-purple-700 text-xs font-medium rounded">
                  Подкатегория: {selectedMessageData.subcategory}
                </span>
              </div>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-6 space-y-4 bg-gray-50">
              {/* User message */}
              <div className="flex gap-3">
                <div
                  className={`w-8 h-8 rounded-full ${VTB_GRADIENT_BR} flex items-center justify-center text-white text-sm font-medium flex-shrink-0`}
                >
                  {selectedMessageData.userAvatar}
                </div>
                <div className="flex-1">
                  <div className="inline-block max-w-[70%] bg-white rounded-2xl rounded-tl-sm p-4 shadow-sm border border-gray-100 whitespace-pre-wrap break-words">
                    <p className="text-gray-900">{selectedMessageData.message}</p>
                  </div>
                  <p className="text-xs text-gray-500 mt-1 ml-1">
                    {formatTime(selectedMessageData.timestamp)}
                  </p>
                </div>
              </div>

              {/* Replies */}
              {selectedMessageData.replies.map((reply) => (
                <div key={reply.id} className="flex gap-3 justify-end">
                  <div className="inline-block max-w-[70%]">
                    <div
                      className={`text-white rounded-2xl rounded-tr-sm p-4 shadow-sm whitespace-pre-wrap break-words ${VTB_GRADIENT_R}`}
                    >
                      <p>{reply.message}</p>
                    </div>
                    <p className="text-xs text-gray-500 mt-1 mr-1 text-right">
                      {formatTime(reply.timestamp)}
                    </p>
                  </div>
                  <div className="w-8 h-8 rounded-full bg-gray-800 flex items-center justify-center text-white text-sm font-medium flex-shrink-0">
                    <User className="w-4 h-4" />
                  </div>
                </div>
              ))}

              {/* Suggested response (основной + раскрывающиеся остальные) */}
              {selectedMessageData.status === 'pending' &&
                selectedMessageData.suggestedResponse && (
                  <div
                    className={`${VTB_LIGHT_GRADIENT_R} border border-[#009FDF]/30 rounded-xl p-4`}
                  >
                    <div className="flex items-start gap-3">
                      <div className="w-8 h-8 bg-[#0A2973] rounded-lg flex items-center justify-center flex-shrink-0">
                        <Sparkles className="w-5 h-5 text-white" />
                      </div>
                      <div className="flex-1">
                        <h3 className="font-semibold text-[#0A2973] mb-2 flex items-center gap-2">
                          Рекомендуемый ответ
                        </h3>

                        {/* основной */}
                        <p className="text-sm text-[#0A2973]/80 mb-3">
                          {selectedMessageData.suggestedResponse}
                        </p>

                        <div className="flex items-center gap-3 flex-wrap">
                          <button
                            onClick={handleSuggestedResponseClick}
                            className="px-4 py-2 bg-[#0A2973] text-white text-sm font-medium rounded-lg hover:opacity-90 transition-colors"
                          >
                            Использовать этот ответ
                          </button>

                          {/* кнопка-стрелочка для раскрытия остальных */}
                          {selectedMessageData.suggestedResponses &&
                            selectedMessageData.suggestedResponses.length > 1 && (
                              <button
                                type="button"
                                onClick={() => setShowAllSuggestions((v) => !v)}
                                aria-expanded={showAllSuggestions}
                                className="inline-flex items-center gap-1 text-[#0A2973] text-sm font-medium hover:opacity-80 transition"
                                title={
                                  showAllSuggestions
                                    ? 'Скрыть дополнительные варианты'
                                    : 'Показать дополнительные варианты'
                                }
                              >
                                {showAllSuggestions
                                  ? 'Скрыть'
                                  : `Ещё ${selectedMessageData.suggestedResponses.length - 1}`}
                                <ChevronDown
                                  className={`w-4 h-4 transition-transform ${showAllSuggestions ? 'rotate-180' : ''}`}
                                />
                              </button>
                            )}
                        </div>

                        {/* остальные варианты с копированием и подкатегориями */}
                        {showAllSuggestions &&
                          selectedMessageData.suggestedResponses &&
                          selectedMessageData.suggestedResponses.length > 1 && (
                            <ul className="mt-4 space-y-2">
                              {selectedMessageData.suggestedResponses.slice(1).map((s, i) => {
                                const absoluteIndex = i + 1;
                                const tag =
                                  selectedMessageData.subcategories?.[absoluteIndex] ??
                                  selectedMessageData.subcategory;
                                const isCopied = copiedIndex === i;

                                return (
                                  <li
                                    key={i}
                                    role="button"
                                    tabIndex={0}
                                    onClick={() => handleCopySuggestion(s, i)}
                                    onKeyDown={(e) => {
                                      if (e.key === 'Enter' || e.key === ' ') {
                                        e.preventDefault();
                                        handleCopySuggestion(s, i);
                                      }
                                    }}
                                    title="Нажмите, чтобы скопировать"
                                    className="
                                      group relative cursor-pointer
                                      rounded-lg border border-[#009FDF]/20 bg-white/80
                                      p-3 transition hover:shadow-md hover:border-[#009FDF]/40
                                      focus:outline-none focus:ring-2 focus:ring-[#009FDF]/40
                                    "
                                  >
                                    {/* тонкая градиентная плашка сверху */}
                                    <div className="h-1 -mt-3 -mx-3 mb-3 rounded-t-lg bg-gradient-to-r from-[#009FDF]/30 to-transparent" />

                                    {/* верхняя строка: подкатегория + кнопка копирования */}
                                    <div className="mb-2 flex items-start justify-between gap-3">
                                      <span className="px-2 py-0.5 rounded bg-purple-100 text-purple-700 text-[11px] font-medium">
                                        {tag}
                                      </span>

                                      <button
                                        type="button"
                                        onClick={(e) => {
                                          e.stopPropagation();
                                          handleCopySuggestion(s, i);
                                        }}
                                        aria-label={isCopied ? 'Скопировано' : 'Скопировать ответ'}
                                        className="
                                          inline-flex items-center gap-1 rounded-md border border-transparent
                                          px-2 py-1 text-xs font-medium
                                          text-[#0A2973] opacity-70
                                          transition hover:opacity-100 hover:border-[#009FDF]/30
                                          focus:outline-none focus:ring-2 focus:ring-[#009FDF]/40
                                        "
                                        title={isCopied ? 'Скопировано' : 'Скопировать'}
                                      >
                                        {isCopied ? (
                                          <>
                                            <Check className="w-4 h-4 text-green-600" />
                                            <span className="text-green-700">Скопировано</span>
                                          </>
                                        ) : (
                                          <>
                                            <Copy className="w-4 h-4" />
                                            <span className="sr-only md:not-sr-only md:inline">
                                              Копировать
                                            </span>
                                          </>
                                        )}
                                      </button>
                                    </div>

                                    {/* текст ответа */}
                                    <p className="leading-relaxed text-sm text-[#0A2973]/90">{s}</p>

                                    {/* бейдж «Скопировано» */}
                                    {isCopied && (
                                      <span
                                        aria-live="polite"
                                        className="absolute -top-2 right-2 text-[10px] px-2 py-0.5 rounded-full bg-green-100 text-green-700 shadow"
                                      >
                                        Скопировано
                                      </span>
                                    )}
                                  </li>
                                );
                              })}
                            </ul>
                          )}
                      </div>
                    </div>
                  </div>
                )}
            </div>

            {/* Reply input */}
            <div className="bg-white border-t border-gray-200 p-4">
              <div className="flex gap-3 items-end">
                <textarea
                  ref={replyRef}
                  value={replyText}
                  onChange={(e) => setReplyText(e.target.value)}
                  onInput={autoResize}
                  onKeyDown={handleKeyDown}
                  aria-label="Поле ответа"
                  placeholder="Введите ответ… (Shift+Enter — перенос, Enter — отправка)"
                  className="flex-1 px-4 py-3 border border-gray-200 rounded-xl resize-none text-gray-900 placeholder-gray-400 leading-relaxed focus:outline-none focus:ring-2 focus:ring-blue-500/40 focus:border-blue-500"
                  rows={1}
                  style={{ height: 'auto' }}
                />

                <button
                  type="button"
                  onClick={handleSendReply}
                  disabled={!replyText.trim() || replyMutation.isPending}
                  title="Отправить (Enter)"
                  aria-busy={replyMutation.isPending}
                  className="
                    inline-flex items-center justify-center gap-2
                    h-12 min-w-[9.5rem] px-5
                    rounded-xl text-white font-medium
                    shadow-sm hover:shadow-md
                    transition-[transform,box-shadow,opacity] duration-150
                    active:scale-[0.99]
                    disabled:opacity-50 disabled:shadow-none disabled:cursor-not-allowed
                    bg-gradient-to-r from-[#009FDF] to-[#0A2973]
                    hover:opacity-90 focus-visible:outline-none
                    focus-visible:ring-2 focus-visible:ring-[#009FDF]/40
                  "
                >
                  {replyMutation.isPending ? (
                    <>
                      <div className="w-5 h-5 border-2 border-white/80 border-t-transparent rounded-full animate-spin" />
                      <span>Отправка…</span>
                    </>
                  ) : (
                    <>
                      <Send className="w-5 h-5" />
                      <span>Отправить</span>
                    </>
                  )}
                </button>
              </div>

              <p className="mt-2 text-xs text-gray-500">
                Shift+Enter — перенос строки · Enter / Ctrl(⌘)+Enter — отправка
              </p>
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-gray-400">
            <div className="text-center">
              <MessageSquare className="w-16 h-16 mx-auto mb-4 opacity-20" />
              <p className="text-lg">Выберите обращение для ответа</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SupportAdminInterface;
