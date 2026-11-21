import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@triton.jit
def add_kernel(x_ptr,y_ptr,output_ptr,number,block_size:tl.constexpr):
    pid=tl.program_id(axis=0)
    start=pid*block_size
    elements=start+tl.arange(0,block_size)

    mask=elements<number

    x=tl.load(x_ptr+elements,mask=mask,other=0.0)
    y=tl.load(y_ptr+elements,mask=mask,other=0.0)

    output_thread_block=x+y

    tl.store(output_ptr+elements,output_thread_block,mask=mask)

def triton_add(x:torch.tensor,y:torch.tensor):
    output=torch.empty_like(x)
    num=output.numel()

    grid= lambda meta: (triton.cdiv(num,meta['block_size']),)
    add_kernel[grid](x,y,output,num,block_size=4)
    return output

if __name__ == "__main__":
    size = 98432  
    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)
    triton_out = triton_add(x, y)
    torch_out = x + y
    print(f"Max difference: {torch.max(torch.abs(triton_out - torch_out)).item()}")
    assert torch.allclose(triton_out, torch_out), "Triton and PyTorch results do not match!"
    print("Verification passed successfully!")
