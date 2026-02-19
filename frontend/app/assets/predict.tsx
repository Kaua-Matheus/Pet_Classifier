import { useRef, useState, type ChangeEvent, type DragEvent } from "react";
import Button from "~/assets/button";

interface Predict {
    filename: string;
    predicted_pet: "Dog" | "Cat";
    pet: {
        pet_confidence: number;
        all_probabilities: {
            Dog: number;
            Cat: number;
        };
    }
    breed: {
        predicted_breed: string
        breed_confidence: number
        breed_probabilities: {}
    }
}

export default function Predict_Card() {
    const apiURL = import.meta.env.VITE_API_URL || "0.0.0.0"
    const apiPORT = import.meta.env.VITE_API_PORT || "8000"

    const [selectedFile, setSelectedFile] = useState<File | null>(null)
    const [preview, setPreview] = useState<string | null>(null)
    const [isLoading, setIsLoading] = useState<boolean>(false)
    const [isDragging, setIsDragging] = useState<boolean>(false)
    const fileInputRef = useRef<HTMLInputElement>(null)
    const [predictResponse, setPredictResponse] = useState<Predict | null>(null)

    const validTypes = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
    const maxSize = 10 * 1024 * 1024

    const validateFile = (file: File): string | null => {
        if (!validTypes.includes(file.type)) {
            return "Tipo de arquivo inválido."
        }
        if (file.size > maxSize) {
            return "Tamanho do arquivo excede o máximo aceito."
        }
        return null
    };

    const handleFileSelect = (file: File) => {
        const err = validateFile(file);
        if (err) {
            alert(err);
            return;
        }

        setSelectedFile(file);
        console.log("Image selecionada: ", file.name);

        const reader = new FileReader();
        reader.onload = (e) => {
            setPreview(e.target?.result as string);
        };
        reader.readAsDataURL(file);
    };

    const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            handleFileSelect(file)
        }
    }

    const handleDragEnter = (e: DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(true);
    }

    const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);
    }

    const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
    }

    const handleDrop = (e: DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);

        if (e.dataTransfer != null) {
            const files = Array.from(e.dataTransfer.files);
            const imageFile = files.find(file => validTypes.includes(file.type));

            if (imageFile) {
                handleFileSelect(imageFile);
            } else {
                alert("Por favor, selecione apenas arquivos de imagem.");
            }
        }
    }

    const handlePredict = async () => {
        if (!selectedFile) return;
        setIsLoading(true);
        setPredictResponse(null);
        
        try {
            const formData = new FormData();
            formData.append("file", selectedFile);

            const response = await fetch(`http://${apiURL}:${apiPORT}/predict/`, {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Erro ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            setPredictResponse(result)

        } catch (err) {
            console.error("Erro na predição: ", err)
        } finally {
            setIsLoading(false)
        }
    }

    const handleClear = () => {
        setSelectedFile(null);
        setPreview(null);
        setPredictResponse(null);
        if (fileInputRef.current) {
            fileInputRef.current.value = "";
        }
    }

    return (
        <div className="w-full max-w-7xl mx-auto">
            {/* Layout Responsivo: Stack vertical em mobile, horizontal em desktop */}
            <div className="flex flex-col lg:flex-row shadow-2xl rounded-2xl overflow-hidden bg-linear-to-br from-gray-900 via-gray-800 to-gray-900">
                
                {/* Upload Section */}
                <div className="w-full lg:w-1/2 bg-linear-to-br from-gray-900 to-gray-800 p-4 sm:p-6 space-y-4 sm:space-y-6 border-b lg:border-b-0 lg:border-r border-gray-700">
                    <div className="text-center">
                        <h2 className="text-xl sm:text-2xl lg:text-3xl font-bold bg-linear-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
                            Fazer Predição
                        </h2>
                        <div className="w-12 sm:w-16 lg:w-20 h-0.5 sm:h-1 bg-linear-to-r from-blue-400 to-purple-500 mx-auto mt-2 rounded-full"></div>
                    </div>
                
                    {/* Área de Drop - Responsiva */}
                    <div
                        className={`
                            relative border-2 border-dashed rounded-xl 
                            p-4 sm:p-6 lg:p-8 
                            text-center cursor-pointer 
                            transition-all duration-300 ease-in-out transform
                            ${isDragging
                                ? "border-blue-500 bg-blue-500/10 scale-105 shadow-lg shadow-blue-500/25"
                                : "border-gray-500 hover:border-gray-400 hover:bg-gray-800/50"
                            }
                            ${selectedFile ? "border-green-500 bg-green-500/10 shadow-lg shadow-green-500/25" : ""}
                            hover:scale-102 hover:shadow-xl
                            before:absolute before:inset-0 before:bg-linear-to-br before:from-blue-500/5 before:to-purple-500/5 before:rounded-xl before:opacity-0 before:transition-opacity before:duration-300
                            hover:before:opacity-100
                        `}
                        onDragEnter={handleDragEnter}
                        onDragLeave={handleDragLeave}
                        onDragOver={handleDragOver}
                        onDrop={handleDrop}
                        onClick={() => fileInputRef.current?.click()}
                    >
                        {preview ? (
                            <div className="space-y-2 sm:space-y-4 relative z-10">
                                <div className="relative group">
                                    <img
                                        src={preview}
                                        alt="image preview"
                                        className="max-w-full max-h-32 sm:max-h-40 lg:max-h-48 mx-auto rounded-lg shadow-2xl transform transition-transform duration-300 group-hover:scale-105"
                                    />
                                    <div className="absolute inset-0 bg-linear-to-t from-black/30 to-transparent rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                                </div>
                                <p className="text-xs sm:text-sm text-gray-300 font-medium break-all px-2">
                                    {selectedFile?.name}
                                </p>
                            </div>
                        ) : (
                            <div className="space-y-3 sm:space-y-4 lg:space-y-6 relative z-10">
                                <div className="text-gray-400">
                                    <div className="relative mx-auto w-10 h-10 sm:w-12 sm:h-12 lg:w-16 lg:h-16 mb-2 sm:mb-4">
                                        <svg
                                            className="w-full h-full"
                                            fill="none"
                                            viewBox="0 0 24 24"
                                            stroke="currentColor"
                                        >
                                            <path
                                                strokeLinecap="round"
                                                strokeLinejoin="round"
                                                strokeWidth={1.5}
                                                d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                                            />
                                        </svg>
                                        <div className="absolute inset-0 bg-linear-to-br from-blue-400 to-purple-500 rounded-full opacity-20 animate-pulse"></div>
                                    </div>
                                </div>
                                <div className="px-2">
                                    <p className="text-gray-300 text-sm sm:text-base lg:text-lg font-medium mb-1 sm:mb-2">
                                        {isDragging ? "Solte a imagem aqui" : (
                                            <>
                                                <span className="hidden sm:inline">Arraste uma imagem ou clique</span>
                                                <span className="sm:hidden">Toque para selecionar</span>
                                            </>
                                        )}
                                    </p>
                                    <p className="text-xs text-gray-500">PNG, JPG, WebP • Máx 10MB</p>
                                </div>
                            </div>
                        )}
                    </div>

                    <input
                        type="file"
                        accept="image/*"
                        className="hidden"
                        onChange={handleFileChange}
                        ref={fileInputRef}
                    />

                    {/* Botões - Responsivos */}
                    <div className="flex flex-col sm:flex-row gap-2 sm:gap-3 justify-center">
                        {selectedFile && (
                            <>
                                <Button
                                    variant="primary"
                                    active={!isLoading}
                                    onClick={handlePredict}
                                    className="flex-1 sm:flex-initial"
                                >
                                    {isLoading ? (
                                        <span className="flex items-center justify-center gap-2">
                                            <svg className="animate-spin h-3 w-3 sm:h-4 sm:w-4" viewBox="0 0 24 24">
                                                <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" className="opacity-25"></circle>
                                                <path fill="currentColor" className="opacity-75" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                            </svg>
                                            <span className="hidden sm:inline">Processando...</span>
                                            <span className="sm:hidden">Analisando...</span>
                                        </span>
                                    ) : (
                                        <>
                                            <span className="hidden sm:inline">Analisar</span>
                                            <span className="sm:hidden">Analisar</span>
                                        </>
                                    )}
                                </Button>
                                <Button
                                    variant="danger"
                                    active={!isLoading}
                                    onClick={handleClear}
                                    className="flex-1 sm:flex-initial"
                                >
                                    <span className="hidden sm:inline">🗑️ Limpar</span>
                                    <span className="sm:hidden">🗑️</span>
                                </Button>
                            </>
                        )}
                    </div>
                </div>

                {/* Results Section - Responsiva */}
                <div className="w-full lg:w-1/2 bg-linear-to-br from-gray-800 to-gray-700 p-4 sm:p-6 space-y-4 sm:space-y-6">
                    <div className="text-center">
                        <h2 className="text-xl sm:text-2xl lg:text-3xl font-bold bg-linear-to-r from-green-400 to-emerald-500 bg-clip-text text-transparent">
                            Resultado
                        </h2>
                        <div className="w-12 sm:w-16 lg:w-20 h-0.5 sm:h-1 bg-linear-to-r from-green-400 to-emerald-500 mx-auto mt-2 rounded-full"></div>
                    </div>

                    <div className="min-h-40 sm:min-h-60 lg:min-h-75 flex items-center justify-center">
                        {isLoading ? (
                            <div className="text-center space-y-3 sm:space-y-4">
                                <div className="relative">
                                    <div className="w-12 h-12 sm:w-16 sm:h-16 mx-auto border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                                    <div className="w-8 h-8 sm:w-12 sm:h-12 mx-auto border-4 border-purple-500 border-b-transparent rounded-full animate-spin absolute top-2 sm:top-2 left-1/2 transform -translate-x-1/2"></div>
                                </div>
                                <p className="text-gray-300 animate-pulse text-sm sm:text-base">Analisando imagem...</p>
                            </div>
                        ) : predictResponse ? (
                            <div className="w-full space-y-4 sm:space-y-6 animate-fadeIn">
                                <div className="bg-linear-to-r from-green-500/20 to-emerald-500/20 rounded-xl p-4 sm:p-6 border border-green-500/30">
                                    <div className="text-center space-y-3 sm:space-y-4">
                                        <div className="text-4xl sm:text-5xl lg:text-6xl">
                                            {predictResponse.predicted_pet === 'Dog' ? '🐕' : '🐱'}
                                        </div>
                                        <div>
                                            <h3 className="text-xl sm:text-2xl font-bold text-white mb-2">
                                                {predictResponse.predicted_pet === 'Dog' ? 'Cachorro' : 'Gato'}
                                            </h3>
                                            <div className="relative">
                                                <div className="bg-gray-600 rounded-full h-2 sm:h-3 overflow-hidden">
                                                    <div 
                                                        className="h-full bg-linear-to-r from-green-400 to-emerald-500 rounded-full transition-all duration-1000 ease-out"
                                                        style={{ width: `${predictResponse.pet.pet_confidence}%` }}
                                                    ></div>
                                                </div>
                                                <p className="text-green-400 font-bold text-base sm:text-lg mt-2">
                                                    {predictResponse.pet.pet_confidence}% de confiança
                                                </p>
                                            </div>
                                        </div>

                                        {/* Info Cards - Stack em mobile */}
                                        <div className="mt-4 sm:mt-6 space-y-3 sm:space-y-4">
                                            <div className="bg-gray-700/50 rounded-lg p-3 sm:p-4 border border-gray-600">
                                                <p className="text-xs text-gray-400 uppercase tracking-wide mb-1 sm:mb-2">Raça Prevista</p>
                                                <p className="text-sm sm:text-xl font-bold text-green-400 wrap-break-word">
                                                    {predictResponse.breed.predicted_breed}
                                                </p>
                                            </div>
                                            <div className="bg-gray-700/50 rounded-lg p-3 sm:p-4 border border-gray-600">
                                                <p className="text-xs text-gray-400 uppercase tracking-wide mb-1 sm:mb-2">Confiança da Raça</p>
                                                <div className="flex items-center gap-2 sm:gap-3">
                                                    <div className="flex-1 bg-gray-600 rounded-full h-1.5 sm:h-2 overflow-hidden">
                                                        <div 
                                                            className="h-full bg-linear-to-r from-blue-400 to-cyan-500 rounded-full transition-all duration-1000 ease-out"
                                                            style={{ width: `${predictResponse.breed.breed_confidence}%` }}
                                                        ></div>
                                                    </div>
                                                    <span className="text-sm sm:text-lg font-bold text-blue-400 min-w-8 sm:min-w-12">
                                                        {predictResponse.breed.breed_confidence}%
                                                    </span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                
                                <div className="text-xs text-gray-400 text-center break-all px-2">
                                    Arquivo: {predictResponse.filename}
                                </div>
                            </div>
                        ) : (
                            <div className="text-center space-y-3 sm:space-y-4 opacity-60 px-4">
                                <div className="text-4xl sm:text-5xl lg:text-6xl animate-pulse">🤖</div>
                                <p className="text-gray-400 text-sm sm:text-base">
                                    <span className="hidden sm:inline">
                                        Insira uma imagem e faça a previsão,<br />
                                        o resultado aparecerá aqui.
                                    </span>
                                    <span className="sm:hidden">
                                        Selecione uma imagem<br />
                                        para ver o resultado
                                    </span>
                                </p>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    )
}