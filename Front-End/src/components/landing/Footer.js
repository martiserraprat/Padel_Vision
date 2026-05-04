export default function Footer() {
  return (
    <footer className="border-t border-white/7 px-8 py-8 flex flex-col gap-1 md:flex-row md:items-center md:justify-between">
      <p className="text-xs text-muted tracking-wide">
        © 2025 <span className="font-display tracking-widest text-cream">PadelVision</span>
        {' '}· Martí Serra Prat & Bernat Domene Solé · Grup 17
      </p>
      <p className="text-xs text-muted tracking-wide">
        Visió per Computador · Escola d'Enginyeria · UAB
      </p>
    </footer>
  )
}
