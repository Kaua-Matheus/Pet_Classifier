import type { ReactNode } from "react"

interface ButtonProps {
    children?: ReactNode,
    onClick?: () => void,
    active?: boolean,
    className?: string,
    variant?: 'primary' | 'secondary' | 'danger',
}

export default function Button(
    { 
        children, 
        onClick, 
        active = true, 
        className,
        variant = "primary"
    }: ButtonProps) {

        const variants = {
        primary: `
            bg-gradient-to-r from-yellow-500 to-orange-600 hover:from-yellow-600 hover:to-orange-700
            text-white shadow-lg hover:shadow-xl
        `,
        secondary: `
            bg-black
            text-white shadow-lg hover:shadow-xl
        `,
        danger: `
            bg-gradient-to-r from-red-500 to-pink-600 hover:from-red-600 hover:to-pink-700
            text-white shadow-lg hover:shadow-xl
        `
    }

    return (
        <button 
            className={`
                relative overflow-hidden
                p-3 px-6 min-w-30 h-12
                rounded-lg font-semibold text-sm
                transition-all duration-300 ease-in-out
                cursor-pointer
                transform active:scale-95
                ${active ? `
                    ${variants[variant]}
                    hover:-translate-y-0.5 hover:scale-102
                    focus:outline-none focus:ring-4 focus:ring-opacity-50
                    ${variant === 'primary' ? 'focus:ring-blue-300' : 
                      variant === 'danger' ? 'focus:ring-red-300' : 
                      'focus:ring-gray-300'}
                ` : `
                    cursor-not-allowed opacity-40 
                    bg-gray-600 text-gray-400
                `}
                before:absolute before:inset-0 before:bg-white before:opacity-0 
                before:transition-opacity before:duration-300
                hover:before:opacity-10
                ${className}
            `}
            onClick={active ? onClick : undefined}
            disabled={!active}
        >
            <span className="relative z-10">{children}</span>
        </button>
    )
}