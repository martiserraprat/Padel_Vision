const features = [
  'Detecció automàtica dels 4 jugadors',
  'Projecció real sobre el pla de la pista',
  'Heatmap individualitzat per jugador',
  'Historial de partits analitzats',
  'Exportació del resultat en imatge',
]

const blobs = [
  { w: 130, h: 90,  color: 'rgba(255,50,50,0.65)',  top: '58%', left: '12%'  },
  { w: 85,  h: 65,  color: 'rgba(255,140,50,0.55)', top: '18%', left: '4%'   },
  { w: 110, h: 75,  color: 'rgba(60,200,60,0.55)',  top: '52%', right: '13%' },
  { w: 75,  h: 55,  color: 'rgba(60,100,255,0.5)',  top: '14%', right: '8%'  },
  { w: 55,  h: 40,  color: 'rgba(255,210,50,0.4)',  top: '38%', left: '38%'  },
]

const players = [
  { label: 'J1', top: '65%', left: '18%' },
  { label: 'J2', top: '22%', left: '8%'  },
  { label: 'J3', top: '58%', right: '18%'},
  { label: 'J4', top: '20%', right: '12%'},
]

function CourtMock() {
  return (
    <div className="rounded-2xl border border-white/5 bg-gray-2 p-6">
      <div
        className="relative w-full overflow-hidden rounded border-2 border-white/15 bg-gray-1"
        style={{ aspectRatio: '3/2' }}
      >
        <div className="absolute inset-y-0 left-1/2 w-px bg-white/10" />
        <div className="absolute inset-x-0 top-1/2 h-px bg-white/5" />

        {blobs.map((b, i) => (
          <div
            key={i}
            className="absolute rounded-full"
            style={{
              width: b.w, height: b.h,
              background: b.color,
              filter: 'blur(22px)',
              top: b.top, left: b.left, right: b.right,
            }}
          />
        ))}

        {players.map((p) => (
          <div
            key={p.label}
            className="absolute flex h-6 w-6 items-center justify-center rounded-full bg-white/15 text-[10px] font-medium border border-white/20 backdrop-blur-sm text-cream"
            style={{ top: p.top, left: p.left, right: p.right, transform: 'translate(-50%, -50%)' }}
          >
            {p.label}
          </div>
        ))}
      </div>

      <div className="mt-4 flex items-center justify-between">
        <span className="text-[11px] uppercase tracking-widest text-muted">Vista de pista 2D</span>
        <span className="rounded-full border border-green/30 bg-green/10 px-3 py-1 text-[10px] text-green">Demo</span>
      </div>
    </div>
  )
}

export default function HeatmapPreview() {
  return (
    <section
      id="features"
      className="border-t border-white/7 px-8 py-24 grid grid-cols-1 gap-16 md:grid-cols-2 md:items-center"
    >
      <div>
        <p className="mb-4 text-[10px] uppercase tracking-[0.25em] text-green">Resultats</p>
        <h2
          className="font-display leading-[0.92] mb-6 text-cream"
          style={{ fontSize: 'clamp(40px,5vw,72px)' }}
        >
          VEUS ON<br />
          <span className="text-green">JUGUES</span><br />
          DE VERITAT
        </h2>
        <p className="text-sm text-muted leading-relaxed mb-8 max-w-sm">
          Cada jugador té el seu propi mapa de calor. Identifica patrons tàctics,
          zones descobertes i oportunitats de millora.
        </p>
        <ul className="flex flex-col gap-3">
          {features.map((f) => (
            <li key={f} className="flex items-center gap-3 text-sm text-muted">
              <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-green" />
              {f}
            </li>
          ))}
        </ul>
      </div>

      <CourtMock />
    </section>
  )
}
