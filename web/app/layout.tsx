import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";
const literata = localFont({
  src: "./fonts/literata-latin.woff2",
  weight: "200 900",
  style: "normal",
  adjustFontFallback: "Times New Roman",
  variable: "--font-literata",
  display: "swap",
});
const work = localFont({
  src: "./fonts/work-sans-latin.woff2",
  weight: "100 900",
  style: "normal",
  variable: "--font-work",
  display: "swap",
});
const mono = localFont({
  src: "./fonts/spline-sans-mono-latin.woff2",
  weight: "400 500",
  style: "normal",
  variable: "--font-mono",
  display: "swap",
});
export const metadata: Metadata = {
  title: "stacks · Find your next chapter",
  description:
    "An independent bookshop powered by transparent recommendations. Explore the shelf, trace every recommendation, and inspect reproducible evaluation.",
  icons: { icon: `${process.env.NEXT_PUBLIC_BASE_PATH ?? ""}/favicon.svg` },
};
export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      className={`${literata.variable} ${work.variable} ${mono.variable}`}
      suppressHydrationWarning
    >
      <body>{children}</body>
    </html>
  );
}
