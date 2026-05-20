'use client'

interface IconProps { s?: number }

export const Icon = {
  Logo: ({ s = 14 }: IconProps) => (
    <svg width={s} height={s} viewBox="0 0 14 14" fill="none">
      <path d="M2 7Q7 2 12 7" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" fill="none"/>
      <path d="M2 8.5Q7 13 12 8.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" fill="none" opacity=".55"/>
      <circle cx="7" cy="7.5" r="0.9" fill="currentColor"/>
    </svg>
  ),
  Sun: ({ s = 14 }: IconProps) => (
    <svg width={s} height={s} viewBox="0 0 14 14" fill="none">
      <circle cx="7" cy="7" r="2.6" stroke="currentColor" strokeWidth="1.2"/>
      <path d="M7 0.8v1.6M7 11.6v1.6M0.8 7h1.6M11.6 7h1.6M2.5 2.5l1.15 1.15M10.35 10.35l1.15 1.15M2.5 11.5l1.15-1.15M10.35 3.65l1.15-1.15" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/>
    </svg>
  ),
  Moon: ({ s = 14 }: IconProps) => (
    <svg width={s} height={s} viewBox="0 0 14 14" fill="none">
      <path d="M11 8.5A4.5 4.5 0 015.5 3a4.5 4.5 0 105.5 5.5z" stroke="currentColor" strokeWidth="1.2" fill="none" strokeLinejoin="round"/>
    </svg>
  ),
  Settings: ({ s = 14 }: IconProps) => (
    <svg width={s} height={s} viewBox="0 0 14 14" fill="none">
      <circle cx="7" cy="7" r="2" stroke="currentColor" strokeWidth="1.2"/>
      <path d="M7 1.5v1.3M7 11.2v1.3M1.5 7h1.3M11.2 7h1.3M3.3 3.3l.9.9M9.8 9.8l.9.9M3.3 10.7l.9-.9M9.8 4.2l.9-.9" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/>
    </svg>
  ),
  Upload: ({ s = 16 }: IconProps) => (
    <svg width={s} height={s} viewBox="0 0 16 16" fill="none">
      <path d="M8 11V3M8 3L5 6M8 3l3 3" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M2 11v1.5A1.5 1.5 0 003.5 14h9a1.5 1.5 0 001.5-1.5V11" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round"/>
    </svg>
  ),
  Play: ({ s = 12 }: IconProps) => (
    <svg width={s} height={s} viewBox="0 0 12 12" fill="none">
      <path d="M2.5 1.5L10 6L2.5 10.5V1.5Z" fill="currentColor"/>
    </svg>
  ),
  Download: ({ s = 14 }: IconProps) => (
    <svg width={s} height={s} viewBox="0 0 14 14" fill="none">
      <path d="M7 1.5V9M7 9L4 6M7 9l3-3" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M2 11v.5A1.5 1.5 0 003.5 13h7a1.5 1.5 0 001.5-1.5V11" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round"/>
    </svg>
  ),
  Reset: ({ s = 14 }: IconProps) => (
    <svg width={s} height={s} viewBox="0 0 14 14" fill="none">
      <path d="M2 7a5 5 0 109-3" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" fill="none"/>
      <path d="M11 1.5v3h-3" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  ),
  ZoomIn: ({ s = 14 }: IconProps) => (
    <svg width={s} height={s} viewBox="0 0 14 14" fill="none">
      <circle cx="6" cy="6" r="4" stroke="currentColor" strokeWidth="1.3" fill="none"/>
      <path d="M9.2 9.2l3.3 3.3M4 6h4M6 4v4" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/>
    </svg>
  ),
  ZoomOut: ({ s = 14 }: IconProps) => (
    <svg width={s} height={s} viewBox="0 0 14 14" fill="none">
      <circle cx="6" cy="6" r="4" stroke="currentColor" strokeWidth="1.3" fill="none"/>
      <path d="M9.2 9.2l3.3 3.3M4 6h4" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/>
    </svg>
  ),
  Move: ({ s = 14 }: IconProps) => (
    <svg width={s} height={s} viewBox="0 0 14 14" fill="none">
      <path d="M7 1.5v11M1.5 7h11M7 1.5L5.5 3M7 1.5L8.5 3M7 12.5L5.5 11M7 12.5L8.5 11M1.5 7L3 5.5M1.5 7L3 8.5M12.5 7L11 5.5M12.5 7L11 8.5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  ),
  Warn: ({ s = 14 }: IconProps) => (
    <svg width={s} height={s} viewBox="0 0 14 14" fill="none">
      <path d="M7 1.5L13 12H1L7 1.5Z" stroke="currentColor" strokeWidth="1.3" fill="none" strokeLinejoin="round"/>
      <line x1="7" y1="5.5" x2="7" y2="8.5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/>
      <circle cx="7" cy="10" r="0.7" fill="currentColor"/>
    </svg>
  ),
  Check: ({ s = 14 }: IconProps) => (
    <svg width={s} height={s} viewBox="0 0 14 14" fill="none">
      <path d="M2.5 7.5L5.5 10.5L11.5 4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" fill="none"/>
    </svg>
  ),
  Close: ({ s = 14 }: IconProps) => (
    <svg width={s} height={s} viewBox="0 0 14 14" fill="none">
      <path d="M3 3l8 8M11 3l-8 8" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round"/>
    </svg>
  ),
  Search: ({ s = 14 }: IconProps) => (
    <svg width={s} height={s} viewBox="0 0 14 14" fill="none">
      <circle cx="6" cy="6" r="4" stroke="currentColor" strokeWidth="1.3" fill="none"/>
      <path d="M9.5 9.5l3 3" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/>
    </svg>
  ),
  ArrowRight: ({ s = 14 }: IconProps) => (
    <svg width={s} height={s} viewBox="0 0 14 14" fill="none">
      <path d="M2 7h10M8 3l4 4-4 4" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  ),
  Json: ({ s = 14 }: IconProps) => (
    <svg width={s} height={s} viewBox="0 0 14 14" fill="none">
      <path d="M5 2H3.5a1.5 1.5 0 00-1.5 1.5v1.7c0 .8-.7 1.3-1 1.3v0c.3 0 1 .5 1 1.3v1.7A1.5 1.5 0 003.5 11H5M9 2h1.5a1.5 1.5 0 011.5 1.5v1.7c0 .8.7 1.3 1 1.3v0c-.3 0-1 .5-1 1.3v1.7A1.5 1.5 0 0110.5 11H9" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" fill="none"/>
    </svg>
  ),
  Link: ({ s = 12 }: IconProps) => (
    <svg width={s} height={s} viewBox="0 0 12 12" fill="none">
      <path d="M5 7L7 5M4.5 8.5L3 10a2 2 0 01-2.8-2.8L1.7 5.7M7.5 3.5L9 2a2 2 0 012.8 2.8L10.3 6.3" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" fill="none"/>
    </svg>
  ),
  Trash: ({ s = 14 }: IconProps) => (
    <svg width={s} height={s} viewBox="0 0 14 14" fill="none">
      <path d="M2.5 3.5h9M5 3.5V2a1 1 0 011-1h2a1 1 0 011 1v1.5M3.5 3.5v8.5a1 1 0 001 1h5a1 1 0 001-1V3.5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" fill="none"/>
    </svg>
  ),
}
