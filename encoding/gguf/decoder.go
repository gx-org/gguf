// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package gguf

import (
	"io"
	"sync"

	"github.com/abrander/gguf"
)

type (
	decoder struct {
		locker  sync.Mutex
		reader  *gguf.Reader
		tensors []TensorInfo
	}

	tensorInfo struct {
		dec *decoder
		ti  gguf.TensorInfo
	}

	reader struct {
		io.Reader
		dec *decoder
	}
)

// ToReaders returns a gguf decoder from a gguf reader.
func ToReaders(reader *gguf.Reader) Readers {
	dec := &decoder{
		reader:  reader,
		tensors: make([]TensorInfo, len(reader.Tensors)),
	}
	for i, ti := range reader.Tensors {
		dec.tensors[i] = &tensorInfo{
			dec: dec,
			ti:  ti,
		}
	}
	return dec
}

// Tensors returns the list of tensors available to the reader.
func (dec *decoder) Tensors() []TensorInfo {
	return dec.tensors
}

// Name of the tensor.
func (ti *tensorInfo) Name() string {
	return ti.ti.Name
}

// Reader of the tensor data. Only a single reader instance can exist at a time.
// If a previous reader has been created, we wait for this previous reader to be
// closed before returning a new reader.
func (ti *tensorInfo) Reader() (io.ReadCloser, error) {
	ti.dec.locker.Lock()
	r, err := ti.ti.Reader()
	if err != nil {
		ti.dec.locker.Unlock()
		return nil, err
	}
	return &reader{dec: ti.dec, Reader: r}, nil
}

// Size is the total size in bytes of the tensor data.
func (ti *tensorInfo) Size() int64 {
	return ti.ti.Size()
}

// Dimensions of the tensor.
func (ti *tensorInfo) Dimensions() []uint64 {
	return ti.ti.Dimensions
}

// Type returns the tensor's GGML type ID.
func (ti *tensorInfo) Type() gguf.GGML {
	return ti.ti.Type
}

// Close the reader and unlock the decoder so that other readers can be created.
func (r *reader) Close() error {
	r.dec.locker.Unlock()
	return nil
}
