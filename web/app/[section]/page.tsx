import Stacks from "@/components/stacks";
const sections = [
  "evaluation",
  "inflation",
  "ope",
  "models",
  "explore",
  "report",
  "trace",
];
export function generateStaticParams() {
  return sections.map((section) => ({ section }));
}
export const dynamicParams = false;
export default async function Section({
  params,
}: {
  params: Promise<{ section: string }>;
}) {
  const { section } = await params;
  return <Stacks page={section} />;
}
