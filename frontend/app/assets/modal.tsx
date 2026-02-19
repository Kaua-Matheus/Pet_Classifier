// React
import { useEffect, useState } from "react"

// Componentes
import Button from "./button";

interface Model {
    class_names: Array<string>;
    num_classes: number;
    model_info: {
        version: string;
        training_date: string;
        accuracy: number;
    };
}

export default function Modal() {
    const apiURL = import.meta.env.VITE_API_URL || "0.0.0.0"
    const apiPORT = import.meta.env.VITE_API_PORT || "8000"

    const [isLoading, setIsLoading] = useState(false)
    const [opened, setOpened] = useState(false)
    const [modelData, setModelData] = useState<Model | undefined>(undefined)

    const handleGetModel = async () => {
        setIsLoading(true)
        try {
            const response = await fetch(`http://${apiURL}:${apiPORT}/predict/`, {
                method: "GET",
            })

            if (!response.ok) {
                throw new Error(`Erro ${response.status}: ${response.statusText}`);
            }

            const data = await response.json()
            setModelData(data)
        } catch (err) {
            console.error("Error trying to get the model info: ", err);
        } finally {
            setIsLoading(false)
        }
    }

    useEffect(() => {
        if (opened) {
            document.body.style.overflow = "hidden";
        } else {
            document.body.style.overflow = "";
        }

        return () => {
            document.body.style.overflow = "";
        };
    }, [opened])

    // Botão do Modal
    function Icon() {
        return (
            <Button
                onClick={() => {
                    if (!opened) {
                        handleGetModel();
                    }
                    setOpened(!opened);
                }}
                variant={opened ? "danger" : "secondary"}
            >
                <span className="hidden sm:inline">
                    {opened ? "Fechar Modal" : "Sobre o Modelo"}
                </span>
                <span className="sm:hidden">
                    {opened ? "Fechar" : "Info"}
                </span>
            </Button>
        )
    }

    function Card() {
        return (
            <>
                {/* Overlay - apenas mobile */}
                <div 
                    className="fixed inset-0 bg-black/60 z-40 md:hidden backdrop-blur-sm"
                    onClick={() => setOpened(false)}
                />
                
                {/* Modal Card */}
                <div className={`
                    fixed z-50 transition-all duration-300 ease-out
                    
                    // Mobile: Modal centralizado e compacto
                    inset-x-4 top-1/2 -translate-y-1/2 
                    max-h-[70vh] w-auto
                    md:inset-auto md:transform-none
                    
                    // Tablet: Canto superior direito, pequeno 
                    md:top-26 md:right-4 md:w-72 md:max-w-[25vw]
                    
                    // Desktop: Ainda menor, canto superior
                    lg:top-30 lg:right-2 lg:w-64 lg:max-w-[20vw]
                    xl:w-70 xl:max-w-[22vw]
                    
                    bg-linear-to-br from-gray-900/95 to-gray-800/95
                    backdrop-blur-md border border-gray-600/50
                    rounded-xl shadow-2xl
                    p-3 sm:p-4 md:p-3
                    
                    overflow-hidden
                    
                    // Animações de entrada
                    animate-in slide-in-from-top-2 fade-in-0
                    md:slide-in-from-right-2
                `}>
                    
                    <div className="flex flex-col h-full max-h-full">
                        
                        {/* Header compacto */}
                        <div className="flex justify-between items-center mb-3 pb-2 border-b border-gray-700/50">
                            <h2 className="text-sm md:text-base font-bold text-white truncate">
                                <span className="hidden sm:inline">Info do Modelo</span>
                                <span className="sm:hidden">Modelo</span>
                            </h2>
                            <button 
                                onClick={() => setOpened(false)}
                                className="text-gray-400 hover:text-white text-lg md:text-base p-1 rounded hover:bg-gray-700/50 transition-colors shrink-0 ml-2"
                            >
                                ✕
                            </button>
                        </div>
                        
                        <div className="flex-1 overflow-y-auto">
                            {isLoading ? (
                                <div className="flex items-center gap-2 justify-center py-4">
                                    <div className="w-3 h-3 border-2 border-pink-500 border-t-transparent rounded-full animate-spin"></div>
                                    <p className="text-xs font-medium text-pink-600">Carregando...</p>
                                </div>
                            ) : modelData ? (
                                <div className="space-y-3">
                                    
                                    {/* Header info compacto */}
                                    <div className="flex justify-between items-center text-xs">
                                        <span className="text-orange-500 font-medium">
                                            v{modelData.model_info.version}
                                        </span>
                                        <span className="text-gray-400">
                                            {modelData.model_info.training_date.slice(0, 10)}
                                        </span>
                                    </div>
                                    
                                    <div className="space-y-2 text-xs">
                                        
                                        {/* Classes */}
                                        <div className="bg-gray-700/30 rounded-lg p-2 border border-gray-600/30">
                                            <div className="flex justify-between items-center">
                                                <span className="text-gray-300 font-medium">Classes:</span>
                                                <div className="flex gap-1">
                                                    {modelData.class_names.map(name => 
                                                        <span key={name} className="text-pink-400">
                                                            {name === 'Dog' || name === 'dog' ? '🐕' : '🐱'}
                                                        </span>
                                                    )}
                                                </div>
                                            </div>
                                        </div>
                                        
                                        {/* Quantidade */}
                                        <div className="bg-gray-700/30 rounded-lg p-2 border border-gray-600/30">
                                            <div className="flex justify-between items-center">
                                                <span className="text-gray-300 font-medium">Total:</span>
                                                <span className="text-pink-400 font-bold">
                                                    {modelData.num_classes}
                                                </span>
                                            </div>
                                        </div>
                                        
                                        {/* Accuracy com barra visual */}
                                        <div className="bg-gray-700/30 rounded-lg p-2 border border-gray-600/30">
                                            <div className="flex justify-between items-center mb-1">
                                                <span className="text-gray-300 font-medium">Accuracy:</span>
                                                <span className="text-green-400 font-bold text-xs">
                                                    {(modelData.model_info.accuracy * 100).toFixed(1)}%
                                                </span>
                                            </div>
                                            {/* Mini barra de progresso */}
                                            <div className="w-full bg-gray-600 rounded-full h-1.5 overflow-hidden">
                                                <div 
                                                    className="h-full bg-linear-to-r from-green-500 to-emerald-400 rounded-full transition-all duration-1000"
                                                    style={{ width: `${modelData.model_info.accuracy * 100}%` }}
                                                ></div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            ) : (
                                <div className="text-center py-4">
                                    <p className="text-red-400 text-xs">
                                        Erro ao carregar
                                    </p>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </>
        )
    }

    return (
        <>
            <Icon />
            {opened && <Card />}
        </>
    )
}