import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Utility Knowledge Assistant",
  description: "A clear starting point for utility knowledge and support.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
