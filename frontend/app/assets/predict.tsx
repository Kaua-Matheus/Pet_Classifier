import { useRef, useState, type ChangeEvent, type DragEvent } from "react";
import Button from "~/assets/button";

interface PredictCardProps {
    onImageSelect?: (file: File) => void;
    onPredict?: (file: File) => void;
}

export default function Predict_Card({ onImageSelect, onPredict }: PredictCardProps) {
    const [selectedFile, setSelectedFile] = useState<File | null>(null)
    const [preview, setPreview] = useState<string | null>(null)
    const [isLoading, setIsLoading] = useState<boolean>(false)
    const [isDragging, setIsDragging] = useState<boolean>(false)
    const fileInputRef = useRef<HTMLInputElement>(null)

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
        onImageSelect?.(file);

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

    // Handlers Drag and Drop
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
        try {
            await onPredict?.(selectedFile);
        } catch (err) {
            console.error("Generic Predict Error: ", err)
        } finally {
            setIsLoading(false)
        }
    }

    const handleClear = () => {
        setSelectedFile(null);
        setPreview(null)
        if (fileInputRef.current) {
            fileInputRef.current.value = "";
        }
    }

    return (
        <div className="max-w-md mx-auto w-[30vw] rounded-lg bg-gray-900 p-4 space-y-5">
            <h2 className="text-2xl font-bold text-center my-6">Fazer Predição</h2>
            
            {/* Area de Drop */}
            <div
                className={`
                    border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all
                    ${isDragging
                        ? "border-blue-500 bg-blue-50"
                        : "border-gray-300 hover:bg-gray-400"
                    }
                    ${selectedFile ? "border-green-500 bg-green-50" : ""}
                    `}
                onDragEnter={handleDragEnter}
                onDragLeave={handleDragLeave}
                onDragOver={handleDragOver}
                onDrop={handleDrop}

                onClick={() => fileInputRef.current?.click()}
            >
                {preview ? (
                    <div className="space-y-4">
                        <img 
                            src={preview} 
                            alt="image preview" 
                            className="max-w-full max-h-48 mx-auto rounded-lg shadow-md"
                        />
                        <p className="text-sm text-gray-600">{selectedFile?.name}</p>
                    </div>
                ) : (
                    <div className="space-y-4">
                        <div className="text-gray-500">
                            <svg 
                                className="mx-auto h-12 w-12 mb-4"
                                fill="none"
                                viewBox="0 0 24 24"
                                stroke="currentColor"
                                >
                                <path 
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth={2}
                                    d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" 
                                />
                            </svg>
                        </div>

                        <div>
                            <p className="text-gray-600">{isDragging ? "Solte a imagem aqui" : "Arraste uma imagem ou clique para selecionar"}</p>
                            <p className="text-xs text-gray-400 mt-2">PNG, JPG, WebP até 10MB</p>
                        </div>
                    </div>
                )}
            </div>

            <div>
                <input 
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={handleFileChange}
                    ref={fileInputRef}
                />
            </div>

            <div className="space-x-2 flex justify-between">
                {selectedFile && (
                    <>
                        <Button 
                            active={selectedFile != null ? true : false}
                            onClick={handlePredict}
                        >
                            {isLoading ? "Processando..." : "Predizer"}
                        </Button>

                        <Button 
                            className="text-red-300" 
                            active={selectedFile != null ? true : false} 
                            onClick={handleClear}
                        >
                            Limpar
                        </Button>
                    </>
                )}
            </div>
        </div>
    )
}