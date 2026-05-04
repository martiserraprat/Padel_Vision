'use client'

export default function Button({
  children,
  variant = 'primary',
  size = 'md',
  onClick,
  className = '',
  type = 'button',
}) {
  const base =
    'inline-flex items-center justify-center rounded-lg font-medium transition-all duration-200 cursor-pointer'

  const variants = {
    primary: 'bg-green text-black hover:bg-green-dark active:scale-[0.98]',
    ghost:   'bg-transparent text-cream border border-white/15 hover:border-white/40 hover:bg-white/5',
    outline: 'bg-transparent text-green border border-green/40 hover:border-green hover:bg-green/5',
  }

  const sizes = {
    sm: 'px-4 py-2 text-xs tracking-wide',
    md: 'px-5 py-2.5 text-sm tracking-wide',
    lg: 'px-8 py-4 text-base',
  }

  return (
    <button type={type} onClick={onClick} className={`${base} ${variants[variant]} ${sizes[size]} ${className}`}>
      {children}
    </button>
  )
}
