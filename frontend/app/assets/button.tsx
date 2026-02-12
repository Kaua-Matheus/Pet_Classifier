import type { ReactNode } from "react"

interface ButtonProps {
    children?: ReactNode,
    onClick?: () => void,
    active?: boolean,
    className?: string,
}

export default function Button({ children, onClick, active=true, className }: ButtonProps) {
    return (
        <button 
            className={`
                p-2 w-36 rounded-sm border border-gray-800
                ${active ? "cursor-pointer text-green-300 font-bold bg-gray-900" : "cursor-not-allowed opacity-50"}
                ` + className
            }
            onClick={onClick}
        >
            { children }
        </button>
    )
}