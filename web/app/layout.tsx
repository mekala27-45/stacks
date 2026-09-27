import type { Metadata } from "next";
import { Literata, Work_Sans, Spline_Sans_Mono } from "next/font/google";
import "./globals.css";
const literata = Literata({
  subsets: ["latin"],
  variable: "--font-literata",
  display: "swap",
});
const work = Work_Sans({
  subsets: ["latin"],
  variable: "--font-work",
  display: "swap",
});
const mono = Spline_Sans_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
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
