export type SupportReply = {
  id: string;
  from: 'support' | 'user';
  message: string;
  timestamp: string;
};

export type SupportMessage = {
  id: string;
  userId: string;
  userName: string;
  userAvatar: string;
  message: string;
  timestamp: string;
  status: 'answered' | 'pending';
  category: string;
  subcategory: string;
  subcategories?: string[];
  suggestedResponse?: string;
  suggestedResponses?: string[];
  priority: 'высокий' | 'средний';
  replies: SupportReply[];
};
