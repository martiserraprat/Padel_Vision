const steps = [
  {
    num:   '01',
    icon:  '📹',
    title: 'Puja el vídeo',
    desc:  'Selecciona un vídeo de la teva partida. Màxim 1 minut, càmera fixa per millors resultats.',
  },
  {
    num:   '02',
    icon:  '⚙️',
    title: 'Anàlisi automàtica',
    desc:  'Substracció de fons, morfologia matemàtica i homografia per detectar i seguir els 4 jugadors.',
  },
  {
    num:   '03',
    icon:  '🔥',
    title: 'Mapa de calor',
    desc:  "Obté un heatmap de les zones on cada jugador ha passat més temps durant el partit.",
  },
]

export default function HowItWorks() {
  return (
    <section id="how" className="border-t border-white/7 px-8 py-24">
      <p className="mb-4 text-[10px] uppercase tracking-[0.25em] text-green">Procés</p>
      <h2 className="font-display leading-none mb-16 text-cream" style={{ fontSize: 'clamp(40px,6vw,80px)' }}>
        COM FUNCIONA
      </h2>

      <div className="grid grid-cols-1 gap-px md:grid-cols-3">
        {steps.map((step, i) => (
          <div
            key={step.num}
            className={`group bg-gray-2 p-10 transition-colors duration-200 hover:bg-gray-3
              ${i === 0 ? 'rounded-t-xl md:rounded-t-none md:rounded-l-xl' : ''}
              ${i === steps.length - 1 ? 'rounded-b-xl md:rounded-b-none md:rounded-r-xl' : ''}`}
          >
            <span className="font-display text-7xl text-green/10 leading-none group-hover:text-green/20 transition-colors duration-200">
              {step.num}
            </span>
            <div className="mt-4 mb-5 flex h-11 w-11 items-center justify-center rounded-xl border border-green/25 bg-green/10 text-xl">
              {step.icon}
            </div>
            <h3 className="mb-3 text-lg font-medium tracking-tight text-cream">{step.title}</h3>
            <p className="text-sm text-muted leading-relaxed">{step.desc}</p>
          </div>
        ))}
      </div>
    </section>
  )
}
