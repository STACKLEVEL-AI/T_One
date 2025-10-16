import React, { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { Send, MessageSquare, Clock, ChevronRight, CheckCircle2 } from 'lucide-react';
import { sendUserMessage } from '../api/support';

const VTB_GRADIENT_R = 'bg-gradient-to-r from-[#009FDF] to-[#0A2973]';
const VTB_LIGHT_BG = 'bg-gradient-to-br from-[#E6F7FF] via-white to-[#EEF3FF]';

const UserSupportInterface: React.FC = () => {
  const [message, setMessage] = useState('');
  const [submitted, setSubmitted] = useState(false);

  const mutation = useMutation({
    mutationFn: sendUserMessage,
    onSuccess: () => {
      setSubmitted(true);
      setMessage('');
      setTimeout(() => setSubmitted(false), 3000);
    },
  });

  const handleSubmit = () => {
    if (message.trim()) {
      mutation.mutate(message);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className={`min-h-screen ${VTB_LIGHT_BG} flex items-center justify-center p-4`}>
      <div className="w-full max-w-2xl">
        <div className="bg-white rounded-2xl shadow-xl overflow-hidden border border-gray-100">
          {/* Header */}
          <div className={`${VTB_GRADIENT_R} p-6 text-white`}>
            <div className="flex items-center gap-3 mb-2">
              <div className="w-12 h-12 bg-white/20 rounded-xl flex items-center justify-center backdrop-blur-sm">
                <MessageSquare className="w-6 h-6" />
              </div>
              <div>
                <h1 className="text-2xl font-bold">Техподдержка</h1>
                <p className="text-white/70 text-sm">Мы всегда готовы помочь</p>
              </div>
            </div>
          </div>

          {/* Content */}
          <div className="p-6">
            {submitted ? (
              <div className="text-center py-12">
                <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  <CheckCircle2 className="w-8 h-8 text-green-600" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">Сообщение отправлено!</h3>
                <p className="text-gray-600">Мы ответим вам в ближайшее время</p>
              </div>
            ) : (
              <>
                <div className="mb-6">
                  <h2 className="text-lg font-semibold text-gray-900 mb-2">
                    Опишите вашу проблему
                  </h2>
                  <p className="text-sm text-gray-600">
                    Наша команда поддержки ответит в течение нескольких минут
                  </p>
                </div>

                <div className="space-y-4">
                  <div className="relative">
                    <textarea
                      value={message}
                      onChange={(e) => setMessage(e.target.value)}
                      onKeyDown={handleKeyDown}
                      placeholder="Введите ваше сообщение… (Enter — отправить, Shift+Enter — новая строка)"
                      className="w-full h-40 px-4 py-3 border border-gray-200 rounded-xl resize-none text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-[#009FDF]/30"
                      disabled={mutation.isPending}
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <button
                      onClick={handleSubmit}
                      disabled={!message.trim() || mutation.isPending}
                      className={`px-6 py-3 ${VTB_GRADIENT_R} text-white rounded-xl font-medium hover:shadow-lg hover:scale-105 transition-all disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 flex items-center gap-2`}
                    >
                      {mutation.isPending ? (
                        <>
                          <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                          Отправка...
                        </>
                      ) : (
                        <>
                          Отправить
                          <Send className="w-4 h-4" />
                        </>
                      )}
                    </button>
                  </div>
                </div>

                {/* FAQ Section */}
                <div className="mt-8 pt-6 border-t border-gray-100">
                  <h3 className="text-sm font-semibold text-gray-900 mb-3">Частые вопросы</h3>
                  <div className="space-y-2">
                    {[
                      'Как стать клиентом банка онлайн?',
                      'Документы для регистрации нового клиента',
                      'Первый вход в Интернет-банк',
                    ].map((faq, i) => (
                      <button
                        key={i}
                        onClick={() => setMessage(faq)}
                        className="w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 rounded-lg transition-colors flex items-center justify-between group"
                      >
                        {faq}
                        <ChevronRight className="w-4 h-4 text-gray-400 group-hover:text-gray-600" />
                      </button>
                    ))}
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default UserSupportInterface;
