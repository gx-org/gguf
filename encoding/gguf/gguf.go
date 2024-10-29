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

// Package gguf unmarshals gguf files into GX structures.
package gguf

import (
	"bytes"
	"io"
	"strconv"
	"strings"
	"unsafe"

	"github.com/pkg/errors"
	"golang.org/x/exp/maps"
	"github.com/gx-org/backend/platform"
	"github.com/gx-org/gx/golang/binder/gobindings/types"
	"github.com/gx-org/gx/golang/encoding"
)

type (
	// Readers is a set of tensor readers.
	Readers interface {
		Tensors() []TensorInfo
	}

	// TensorInfo reads the data of a given tensor.
	TensorInfo interface {
		// Name of the tensor.
		Name() string

		// Reader of the tensor data.
		Reader() (io.ReadCloser, error)

		// Size is the total size in bytes of the tensor data.
		Size() int64

		// Dimensions of the tensor.
		Dimensions() []uint64
	}

	keyValue struct {
		parent *keyValue
		name   string

		path []string
		info TensorInfo
	}

	walkerStruct struct{ set []keyValue }
	walkerSlice  struct{ set map[int][]keyValue }
	walkerData   struct{ set []keyValue }
)

func (kv *keyValue) child(name string) keyValue {
	return keyValue{
		parent: kv,
		name:   name,

		path: kv.path[1:],
		info: kv.info,
	}
}

func (kv *keyValue) fullPath() string {
	full := ""
	if kv.parent != nil {
		full = kv.parent.fullPath()
	}
	return full + kv.name
}

func selectChildren(set []keyValue, name string) []keyValue {
	children := []keyValue{}
	for _, kv := range set {
		path := kv.path
		if len(path) == 0 {
			continue
		}
		if path[0] != name {
			continue
		}
		children = append(children, kv.child(name))
	}
	return children
}

func keys(set []keyValue) []string {
	keys := map[string]bool{}
	for _, kv := range set {
		first := ""
		if len(kv.path) > 0 {
			first = kv.path[0]
		}
		keys[first] = true
	}
	return maps.Keys(keys)
}

func (w walkerStruct) Field(name string) (encoding.Data, error) {
	child := walkerData{set: selectChildren(w.set, name)}
	if len(child.set) == 0 {
		return nil, errors.Errorf("no field named %q (in %v)", name, keys(w.set))
	}
	return child, nil
}

func (walkerStruct) TagName() string {
	return "gguf"
}

func (w walkerSlice) Len() int {
	return len(w.set)
}

func (w walkerSlice) Index(i int) (encoding.Data, error) {
	children, ok := w.set[i]
	if !ok {
		return nil, errors.Errorf("no element at index %d", i)
	}
	return walkerData{set: children}, nil
}

func (w walkerData) ToDataSlice() (_ encoding.DataSlice, err error) {
	sl := walkerSlice{set: make(map[int][]keyValue)}
	for _, kv := range w.set {
		if len(kv.path) == 0 {
			return nil, errors.Errorf("element in slice has no value")
		}
		first := kv.path[0]
		i, err := strconv.Atoi(first)
		if err != nil {
			return nil, errors.Errorf("cannot build slice for key %d: %v", i, err)
		}
		sl.set[i] = append(sl.set[i], kv.child(first))
	}
	return sl, nil
}

func (w walkerData) ToDataStruct() (encoding.DataStruct, error) {
	return walkerStruct{set: w.set}, nil
}

type arrayFuture struct {
	kv keyValue
}

func (pr arrayFuture) Value() (types.ArrayBridge, error) {
	tensor := pr.kv.info
	r, err := tensor.Reader()
	if err != nil {
		return nil, err
	}
	defer r.Close()
	const float32ByteSize = 4
	buf := bytes.Buffer{}
	n, err := io.CopyN(&buf, r, tensor.Size())
	if err != nil {
		return nil, err
	}
	if n != tensor.Size() {
		return nil, errors.Errorf("not enough bytes written: got %d but want %d", n, tensor.Size())
	}
	// GGUF stores dimensions in minor-to-major order.
	dims := tensor.Dimensions()
	axes := make([]int, len(dims))
	length := 1
	for i, dim := range dims {
		axes[len(dims)-1-i] = int(dim)
		length *= int(dim)
	}
	if length != int(tensor.Size())/float32ByteSize {
		return nil, errors.Errorf("mismatch between the axis (%v=%d elements) and the size of the buffer (%d/%d=%d elements)",
			axes, length, tensor.Size(), float32ByteSize, tensor.Size()/float32ByteSize,
		)
	}
	flat := unsafe.Slice((*float32)(unsafe.Pointer(&buf.Bytes()[0])), length)
	return types.ArrayFloat32(flat, axes...), nil
}

func (w walkerData) ValueFuture() (encoding.ValueFuture, error) {
	if len(w.set) == 0 {
		return nil, errors.Errorf("empty leaf")
	}
	if len(w.set) > 1 {
		return nil, errors.Errorf("not a leaf")
	}
	return arrayFuture{kv: w.set[0]}, nil
}

// UnmarshalOnDevice populates a GX structure from GGUF readers.
func UnmarshalOnDevice(dev platform.Device, target types.Bridger, readers Readers) error {
	w := walkerData{}
	for _, info := range readers.Tensors() {
		w.set = append(w.set, keyValue{
			path: strings.Split(info.Name(), "."),
			info: info,
		})
	}
	return encoding.Unmarshal(encoding.SendToDevice(dev), target, w)
}
