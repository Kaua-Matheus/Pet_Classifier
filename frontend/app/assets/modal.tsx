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

    const apiURL = import.meta.env.VITE_API_URL || "localhost"
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

    // Logo
    function Icon() {
        return (
            <>
                {
                    !opened
                    ? (
                        <Button
                        onClick={() => {handleGetModel(); setOpened(!opened)}}
                        variant="secondary">
                            Sobre o Modelo
                        </Button>
                    )
                    : (
                        <Button
                        onClick={() => setOpened(!opened)}
                        variant="danger">
                            Fechar Modal
                        </Button>
                    )
                }
            </>
        )
    }

    // Card aberto
    function Card() {
        return (
                <div className="
                    fixed top-18 left-2 h-[22vh] w-[25vh] 
                    bg-linear-to-r from-gray-900 to-gray-800 rounded-lg
                    px-6 py-8
                    ">
                    
                    <div>
                        <div className="flex flex-col justify-center">
                            <h2 className="flex text-lg font-bold mb-4">Detalhes do modelo</h2>
                            {
                                isLoading && modelData != undefined
                                ? (<p className="text-md font-bold text-pink-600">Carregando...</p>) 
                                : (<div>
                                    <div className="font-semibold flex text-orange-600 justify-between">
                                        <p>V{modelData?.model_info.version}</p>
                                        <p>{modelData?.model_info.training_date.slice(0, 9)}</p>
                                    </div>
                                    <br />
                                    <p className="text-sm font-semibold">Classes: <span className="text-pink-600 font-normal">{(modelData)?.class_names.map(name => name).join(", ")}</span></p>
                                    <p className="text-sm font-semibold">Qtd Classes: <span className="text-pink-600 font-normal">{modelData?.num_classes}</span></p>
                                    <p className="text-sm font-semibold">Accuracy: <span className="text-pink-600 font-normal">{(modelData?.model_info.accuracy)?.toFixed(4)}</span></p>
                                </div>)
                            }
                        </div>
                    </div>
                </div>
        )
    }

    return (
        <>
            <Icon />

            {opened && (
                Card()
            )}
        </>
    )
}