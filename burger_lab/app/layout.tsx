import type { Metadata } from "next";
import { headers } from "next/headers";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host =
    requestHeaders.get("x-forwarded-host") ??
    requestHeaders.get("host") ??
    "localhost:3000";
  const protocol =
    requestHeaders.get("x-forwarded-proto") ??
    (host.startsWith("localhost") ? "http" : "https");
  const metadataBase = new URL(`${protocol}://${host}`);

  return {
    metadataBase,
    title: "NEXUS · 多主体汉堡协作训练平台",
    description:
      "基于 Overcooked 与 MAPPO/CTDE 的多智能体分工协作仿真、评测与失败回流演示。",
    icons: {
      icon: "/favicon.ico",
      shortcut: "/favicon.ico",
    },
    openGraph: {
      title: "NEXUS · Multi-Agent Emergence Lab",
      description: "局部感知、集中训练、去中心化执行的汉堡协作任务平台。",
      type: "website",
      images: [
        {
          url: new URL("/og.png", metadataBase),
          width: 1200,
          height: 630,
          alt: "NEXUS 多主体汉堡协作训练平台",
        },
      ],
    },
    twitter: {
      card: "summary_large_image",
      title: "NEXUS · Multi-Agent Emergence Lab",
      description: "MAPPO / CTDE 多主体协作仿真与 failure-case 学习闭环。",
      images: [new URL("/og.png", metadataBase)],
    },
  };
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body className={`${geistSans.variable} ${geistMono.variable}`}>
        {children}
      </body>
    </html>
  );
}
