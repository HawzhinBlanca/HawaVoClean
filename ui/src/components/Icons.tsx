import type { SVGProps } from 'react';

type P = SVGProps<SVGSVGElement> & { size?: number };

function base(size: number | undefined, rest: SVGProps<SVGSVGElement>): SVGProps<SVGSVGElement> {
  const s = size ?? 16;
  return {
    width: s,
    height: s,
    viewBox: '0 0 16 16',
    fill: 'none',
    stroke: 'currentColor',
    strokeWidth: 1.5,
    strokeLinecap: 'round',
    strokeLinejoin: 'round',
    'aria-hidden': true,
    focusable: false,
    ...rest,
  };
}

export function IconPlay({ size, ...rest }: P) {
  return (
    <svg {...base(size, rest)}>
      <path d="M4.5 3.2v9.6L12.5 8z" fill="currentColor" stroke="none" />
    </svg>
  );
}

export function IconPause({ size, ...rest }: P) {
  return (
    <svg {...base(size, rest)}>
      <rect x="3.5" y="3" width="3" height="10" rx="0.8" fill="currentColor" stroke="none" />
      <rect x="9.5" y="3" width="3" height="10" rx="0.8" fill="currentColor" stroke="none" />
    </svg>
  );
}

export function IconFolder({ size, ...rest }: P) {
  return (
    <svg {...base(size, rest)}>
      <path d="M2 4.5A1.5 1.5 0 0 1 3.5 3h2.6l1.4 1.5h5A1.5 1.5 0 0 1 14 6v5.5A1.5 1.5 0 0 1 12.5 13h-9A1.5 1.5 0 0 1 2 11.5z" />
      <path d="M2 7h12" />
    </svg>
  );
}

export function IconClip({ size, ...rest }: P) {
  return (
    <svg {...base(size, rest)}>
      <rect x="1.75" y="3.75" width="12.5" height="8.5" rx="1.25" />
      <path d="M4.25 3.75v8.5M11.75 3.75v8.5M1.75 6.5h2.5M11.75 6.5h2.5M1.75 9.5h2.5M11.75 9.5h2.5" />
      <path d="M6.75 6.5v3l2.5-1.5z" fill="currentColor" stroke="none" />
    </svg>
  );
}

export function IconDrop({ size, ...rest }: P) {
  return (
    <svg {...base(size, rest)}>
      <path d="M8 2.5v7M5.25 7l2.75 2.75L10.75 7" />
      <path d="M2.75 10.5v1.25A1.25 1.25 0 0 0 4 13h8a1.25 1.25 0 0 0 1.25-1.25V10.5" />
    </svg>
  );
}

export function IconReveal({ size, ...rest }: P) {
  return (
    <svg {...base(size, rest)}>
      <circle cx="7" cy="7" r="4.25" />
      <path d="M10.2 10.2 13.5 13.5" />
    </svg>
  );
}

export function IconImport({ size, ...rest }: P) {
  return (
    <svg {...base(size, rest)}>
      <path d="M8 2.5v7.25M5.25 7.25 8 10l2.75-2.75" />
      <path d="M2.5 10.75V12a1.5 1.5 0 0 0 1.5 1.5h8A1.5 1.5 0 0 0 13.5 12v-1.25" />
    </svg>
  );
}

export function IconReplace({ size, ...rest }: P) {
  return (
    <svg {...base(size, rest)}>
      <path d="M3 6h9l-2.5-2.5M13 10H4l2.5 2.5" />
    </svg>
  );
}

export function IconCancel({ size, ...rest }: P) {
  return (
    <svg {...base(size, rest)}>
      <path d="M4.5 4.5 11.5 11.5M11.5 4.5 4.5 11.5" strokeWidth="1.75" />
    </svg>
  );
}

export function IconCheck({ size, ...rest }: P) {
  return (
    <svg {...base(size, rest)}>
      <path d="M3.5 8.5 6.5 11.5 12.5 5" strokeWidth="1.75" />
    </svg>
  );
}

export function IconWarn({ size, ...rest }: P) {
  return (
    <svg {...base(size, rest)}>
      <path d="M8 2.5 14 13H2z" />
      <path d="M8 6.5v3M8 11.4v.1" strokeWidth="1.75" />
    </svg>
  );
}

export function IconSpark({ size, ...rest }: P) {
  return (
    <svg {...base(size, rest)}>
      <path d="M8 2v3M8 11v3M2 8h3M11 8h3M4 4l2 2M10 10l2 2M12 4l-2 2M6 10l-2 2" />
    </svg>
  );
}

export function IconBolt({ size, ...rest }: P) {
  return (
    <svg {...base(size, rest)}>
      <path d="M9 1.75 3.5 9h4l-.5 5.25L12.5 7h-4z" fill="currentColor" stroke="none" />
    </svg>
  );
}

export function IconRetry({ size, ...rest }: P) {
  return (
    <svg {...base(size, rest)}>
      <path d="M2.5 8a5.5 5.5 0 1 0 1.2-3.4M2.5 3v3h3" />
    </svg>
  );
}

