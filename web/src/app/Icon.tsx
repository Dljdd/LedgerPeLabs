interface IconProps {
  name: "arrow" | "check" | "chevron" | "pause" | "play" | "reset" | "step" | "warning";
  size?: number;
}

const paths: Record<IconProps["name"], React.ReactNode> = {
  arrow: <path d="M5 12h13m-5-5 5 5-5 5" />,
  check: <path d="m5 12 4 4L19 6" />,
  chevron: <path d="m9 18 6-6-6-6" />,
  pause: <path d="M8 5v14m8-14v14" />,
  play: <path d="m8 5 11 7-11 7Z" />,
  reset: <path d="M4 4v6h6M5.6 16a8 8 0 1 0 .4-9.7L4 10" />,
  step: <path d="m6 5 9 7-9 7Zm11 0v14" />,
  warning: <path d="M12 9v4m0 4h.01M4.9 20h14.2a2 2 0 0 0 1.73-3L13.73 4.7a2 2 0 0 0-3.46 0L3.17 17a2 2 0 0 0 1.73 3Z" />,
};

export function Icon({ name, size = 18 }: IconProps) {
  return (
    <svg aria-hidden="true" className="icon" fill="none" height={size} viewBox="0 0 24 24" width={size}>
      <g stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.8">
        {paths[name]}
      </g>
    </svg>
  );
}
